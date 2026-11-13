#pragma once

// Mega MoE across NVLink domains: the constants of one build and the per-launch state every role starts from.
//
// Vocabulary. A node is one NVLink domain. Every GPU has one counterpart on each other node (the GPU with the
// same index there); all traffic between two nodes runs between counterparts. A GPU is at once
//   a source    of its tokens: it puts its inputs into a mirror on each counterpart,
//   a proxy     for its counterparts: hosts on its node pull the mirrored rows from it, write their partial
//               results to it, and it returns one summed record per token,
//   a host      of its own experts, exactly as in the stock kernel.
// "Remote node slot" nr in [0, kR) enumerates the other nodes in node order.

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/comm/gin.cuh>
#include <deep_gemm/comm/landing.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/mega_moe_ib.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

namespace deep_gemm::ib {

// The plan of this build, as the host printed it (the only place the macro is read)
struct BuildPlan {
    static constexpr layout::IBPlan value = DG_MEGA_MOE_IB_PLAN;
};

template <uint32_t kNumRanks, uint32_t kNumSMs,
          uint32_t kNumMaxTokens, uint32_t kHidden, uint32_t kNumTopk, uint32_t kNumExperts,
          typename plan_t = BuildPlan>
struct Geometry {
    static constexpr layout::IBPlan kPlan = plan_t::value;
    static constexpr bool kDiag = kPlan.diagnostics > 0;
    static constexpr bool kSenderTables = kPlan.routing == layout::IBRouting::SenderTables;

    static constexpr uint32_t kT = kNumMaxTokens, kH = kHidden, kK = kNumTopk;
    // Zero-network measurement: the first two launches fill both buffer sets over the real network; from the
    // third on nothing is put, nothing the NIC wrote is re-armed and epochs in landed headers are not checked
    // (valid only for identical inputs every launch)
    static constexpr bool kRearm = not kPlan.zero_network;
    CUTLASS_DEVICE static bool is_network_live(const uint64_t& epoch) { return not (kPlan.zero_network and epoch > 2); }

    // Topology
    static constexpr uint32_t kLsa = kPlan.ranks_per_node;
    static constexpr uint32_t kNumNodes = kNumRanks / kLsa;
    static constexpr uint32_t kR = kNumNodes - 1;
    static constexpr uint32_t kNumRemoteSources = kNumRanks - kLsa;
    static constexpr uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks;
    DG_STATIC_ASSERT(kLsa > 0 and kNumRanks % kLsa == 0, "Ranks must fill whole nodes");
    DG_STATIC_ASSERT(kNumNodes <= layout::MegaMoEIBSignals::kMaxNodes and
                     kNumRemoteSources <= layout::MegaMoEIBSignals::kMaxSources, "Too many nodes or ranks");
    DG_STATIC_ASSERT(not kPlan.first_need_order or kSenderTables, "First-need order needs sender tables");

    CUTLASS_DEVICE static uint32_t node_of(const uint32_t& rank) { return rank / kLsa; }
    CUTLASS_DEVICE static uint32_t counterpart(const uint32_t& rank, const uint32_t& node) { return node * kLsa + rank % kLsa; }
    CUTLASS_DEVICE static uint32_t node_of_slot(const uint32_t& nr, const uint32_t& my_node) { return nr < my_node ? nr : nr + 1; }
    CUTLASS_DEVICE static uint32_t slot_of_node(const uint32_t& node, const uint32_t& my_node) { return node < my_node ? node : node - 1; }
    // Remote sources as a receiver numbers them: by remote node slot, then by index in the node
    CUTLASS_DEVICE static uint32_t source_of(const uint32_t& rank, const uint32_t& my_node) {
        return slot_of_node(node_of(rank), my_node) * kLsa + rank % kLsa;
    }
    CUTLASS_DEVICE static uint32_t rank_of_source(const uint32_t& k, const uint32_t& my_node) {
        return node_of_slot(k / kLsa, my_node) * kLsa + k % kLsa;
    }

    // Buffer geometry (shared with the host) and the units of the two transfers
    // (a function: device code cannot take the address of a static constexpr object)
    CUTLASS_HOST_DEVICE static constexpr layout::IBBufferLayout buf() { return {kNumMaxTokens, kHidden, kNumTopk, kNumExperts}; }
    static constexpr uint32_t kNumRecordsPerPut = 32;   // records per return put
    static constexpr uint32_t kNumRowsPerPut = kPlan.rows_per_put;     // token rows per forward put
    static constexpr uint32_t kNumRowIssuers = kPlan.row_issuers;      // posting threads (SMs) per remote node
    DG_STATIC_ASSERT(kNumRowsPerPut > 0 and kNumRowIssuers > 0, "Invalid row posting plan");
    static constexpr uint32_t kNumMaxChunks = math::constexpr_ceil_div(kNumMaxTokens, kNumRecordsPerPut);
    DG_STATIC_ASSERT(kNumMaxChunks <= layout::MegaMoEIBSignals::kMaxFwdChunks, "Too many return chunks");
    DG_STATIC_ASSERT(buf().record_bytes() % 16 == 0 and kHidden % 512 == 0, "Rows and records are read in 16-byte units, a warp at a time");

    // Which SM does what before the first GEMM task (epilogue warps; they are idle until then).
    //   SM 0                    posts the meta (top-k indices, weights, header) to every remote node
    //   SMs [1, 1 + kR x I)     post the scale factors and token rows, I = kNumRowIssuers SMs per remote node
    //   the last kNumReceiverSMs  receive one remote source each (slice): its counts and slot table; the last kR
    //                           of them are the sources this GPU is the proxy of, and also do the proxy intake
    // An SM's dispatch warps wait for its epilogue warps before pulling, so SMs that are busy at the start take
    // no part in the pull rotation.
    static constexpr uint32_t kMetaSM = 0;
    static constexpr uint32_t kFirstRowSM = 1;
    static constexpr uint32_t kNumReceiverSMs = kNumRemoteSources * (kSenderTables ? 1 : kPlan.scan_slices);
    static constexpr uint32_t kFirstReceiverSM = kNumSMs - kNumReceiverSMs;
    static constexpr uint32_t kFirstProxySM = kNumSMs - kR;
    static constexpr uint32_t kFirstPullSM = kFirstRowSM + kR * kNumRowIssuers;
    static constexpr uint32_t kEndPullSM = kSenderTables ? kFirstProxySM : kFirstReceiverSM;
    DG_STATIC_ASSERT(kFirstPullSM < kEndPullSM and kEndPullSM <= kNumSMs, "Not enough SMs for the start roles");

    // The remote source a receiver SM handles: the last kR SMs take this GPU's counterparts (node slot order from
    // the end), the SMs before them the other ranks of the remote nodes
    CUTLASS_DEVICE static uint32_t source_of_receiver(const uint32_t& sm_idx, const uint32_t& my_rank) {
        const uint32_t rev = kNumSMs - 1 - sm_idx, my_idx = my_rank % kLsa;
        if (rev < kR)
            return rev * kLsa + my_idx;
        const uint32_t q = rev - kR, j = q % (kLsa - 1);
        return (q / (kLsa - 1)) * kLsa + (j < my_idx ? j : j + 1);
    }

    // Tables of this GPU inside `MegaMoEIBSignals::tables`, [remote node slot][token] each
    static constexpr uint32_t kTableT = kNumMaxTokens;
    static constexpr uint32_t kOffReturnPerm = 0;                       // proxy: return position -> token
    static constexpr uint32_t kOffReturnPos = kOffReturnPerm + kR * kTableT;  // source: token -> return position
    static constexpr uint32_t kOffNeedPos = kOffReturnPos + kR * kTableT;     // source: token -> mirror row
    static constexpr uint32_t kOffNeedInv = kOffNeedPos + kR * kTableT;       // source: mirror row -> token
    static constexpr uint32_t kOffSortKeys = kOffNeedInv + kR * kTableT;      // sort keys, [SM][token]
    static constexpr uint32_t kOffSortRows = kOffSortKeys + kNumSMs * kTableT;
    DG_STATIC_ASSERT(kOffSortRows <= layout::MegaMoEIBSignals::kNumTableWords / 2, "IB tables do not fit");
};

// Per-launch state of a thread: where things are and which launch this is
struct Launch {
    uint8_t* tail;                       // start of the IB tail of this GPU's symmetric buffer
    layout::MegaMoEIBSignals* signals;
    layout::IBHandles handles;
    int64_t base;                        // this GPU's symmetric buffer (Gin window base)
    uint64_t epoch;                      // counts launches of this GPU from 1
    uint32_t parity;                     // epoch & 1: which copy of every NIC-written region this launch uses
    uint32_t rank, my_node;
    bool network;                        // false only in a zero-network measurement launch

    template <uint32_t kNumRanks, typename geometry_t>
    CUTLASS_DEVICE Launch(const layout::SymBuffer<kNumRanks>& sym_buffer, const layout::IBHandles& handles,
                          void* stock_signals, const int64_t& num_stock_bytes, const geometry_t&):
        tail(math::advance_ptr<uint8_t>(sym_buffer.get_base_ptr(), layout::IBBufferLayout::tail_offset(num_stock_bytes))),
        signals(layout::IBBufferLayout::signals(stock_signals, tail)),
        handles(handles), base(sym_buffer.base),
        epoch(ptx::ld_volatile(&signals->epoch) + 1), parity(static_cast<uint32_t>(epoch & 1)),
        rank(sym_buffer.rank_idx), my_node(geometry_t::node_of(sym_buffer.rank_idx)),
        network(geometry_t::is_network_live(epoch)) {}

    CUTLASS_DEVICE comm::landing::Site site(const char* name, const uint32_t& a = 0, const uint32_t& b = 0) const {
        return {name, rank, a, b};
    }
};

// The Gin context of a put, by the plan's assignment (comm/gin.cuh)
template <typename G>
CUTLASS_DEVICE comm::Gin make_gin(const Launch& launch, const comm::Stream& stream, const uint32_t& nr, const uint32_t& lane = 0) {
    if constexpr (G::kPlan.contexts_per_node)
        return comm::Gin(launch.handles, launch.base, comm::Gin::context_of(stream, lane, nr, G::kR), launch.network);
    else
        return comm::Gin(launch.handles, launch.base, comm::Gin::context_of(stream, stream == comm::Stream::Return ? lane + nr : lane), launch.network);
}

// Node slot `node` of this GPU's tail, and the regions in it (see layout::IBBufferLayout)
template <typename geometry_t>
struct Slots {
    using G = geometry_t;
    CUTLASS_DEVICE static uint8_t* node(const Launch& launch, const uint32_t& n) { return launch.tail + G::buf().off_node(n); }
    CUTLASS_DEVICE static uint8_t* mirror(const Launch& launch, const uint32_t& n) { return node(launch, n) + G::buf().off_mirror(launch.parity); }
    CUTLASS_DEVICE static uint8_t* ret(const Launch& launch, const uint32_t& n) { return node(launch, n) + G::buf().off_ret(launch.parity); }
    CUTLASS_DEVICE static uint8_t* proxy_combine(const Launch& launch, const uint32_t& n) { return node(launch, n) + G::buf().off_proxy_combine(); }
    CUTLASS_DEVICE static uint8_t* fwd_stage(const Launch& launch, const uint32_t& n) { return node(launch, n) + G::buf().off_fwd_stage(); }
    // The slot of this GPU's own node is never written by a NIC. Bytes at offset o of it are put to offset o of
    // the peer's slot for this node, so it is the address of every forward put ("outbox") and free scratch:
    //   combine_scratch  the last T x H x 4 bytes of its proxy-combine region: fp32 sums of the source combine
    CUTLASS_DEVICE static uint8_t* outbox_mirror(const Launch& launch) { return mirror(launch, launch.my_node); }
    CUTLASS_DEVICE static uint8_t* outbox_ret(const Launch& launch) { return ret(launch, launch.my_node); }
    CUTLASS_DEVICE static float* combine_scratch(const Launch& launch) {
        return reinterpret_cast<float*>(proxy_combine(launch, launch.my_node) + G::buf().proxy_combine_stride()
                                        - static_cast<int64_t>(G::buf().num_max_tokens) * G::buf().hidden * 4);
    }
    DG_STATIC_ASSERT(static_cast<int64_t>(G::kR) * G::buf().num_max_tokens * G::buf().hidden
                     + G::buf().num_max_tokens * G::buf().hidden * 4 <= G::buf().proxy_combine_stride() or not G::kPlan.first_need_order,
                     "Combine scratch must not reach the staged rows at the start of the own-node proxy-combine region");
};

} // namespace deep_gemm::ib
