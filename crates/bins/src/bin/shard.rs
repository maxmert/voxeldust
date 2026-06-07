//! THE shard binary (HR3: one binary; shard types are `ShardProfile` configs —
//! P1 runs the stub profile). A thin shell: config → mesh → build_app → register
//! → tick loop. All logic lives in the tested libs.

use vd_io_prod::mesh::{MeshConfig, spawn_mesh};
use vd_io_prod::runtime::{EnvConfig, TickPacer};
use vd_io_prod::trust::ClusterTrust;
use vd_node::app::{NodeConfig, build_app};
use vd_node::follower::register_clock_follower;
use vd_sim::capability::NodeKind;
use vd_sim::stub::{StubConfig, register_stub_shard};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt().with_env_filter("info").init();
    let env = EnvConfig::from_process_env();
    let local = env.node_id("VD_NODE_ID")?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?;
    let trust = ClusterTrust::from_der_dir(std::path::Path::new(&env.string("VD_TRUST_DIR")?))?;
    let (transport, _control) = spawn_mesh(
        runtime.handle(),
        &trust,
        &MeshConfig::new(
            local,
            env.parse("VD_BIND")?,
            env.peer_book("VD_PEERS")?,
            env.parse("VD_OUTBOUND_CAP")?,
        ),
    )?;
    let mut node = build_app(
        NodeConfig {
            node_id: local,
            kind: NodeKind::StubShard,
        },
        transport,
    );
    let (world, schedule) = node.parts_mut();
    register_clock_follower(world, schedule);
    let realm_seed: u64 = env.parse("VD_REALM_SEED")?;
    register_stub_shard(
        world,
        schedule,
        StubConfig {
            realm: vd_core::pose::RealmId::System(realm_seed),
            frame: vd_core::pose::FrameRef::SystemSpace {
                system_seed: realm_seed,
            },
            move_speed_mps: env.parse("VD_SPEED")?,
            tick_dt_s: env.parse("VD_TICK_DT")?,
            orchestrator: env.node_id("VD_ORCH")?,
            mint_seed: env.parse("VD_MINT_SEED")?,
            // Bounded in production: the input-conservation log is a metrics ring,
            // not an unbounded audit trail (SCALE-3).
            input_log_capacity: env.parse("VD_INPUT_LOG_CAP")?,
        },
    );
    let mut pacer = TickPacer::new(env.parse("VD_TICK_HZ")?);
    loop {
        let _ = node.step_tick();
        let _ = pacer.wait();
    }
}
