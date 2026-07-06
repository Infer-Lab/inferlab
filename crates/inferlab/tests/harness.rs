//! Targeted tests for the shared test-harness machinery itself.
//! Every test runs against a private registry or
//! lease directory, so concurrent executions cannot interfere with each
//! other or with real suite runs sweeping the shared machine registry.

mod support;

use std::error::Error;
use std::fs;
use std::os::unix::process::CommandExt;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::process::{Child, Command};

/// A detached process-group leader standing in for a fixture server. The
/// handle keeps the `Child` so the zombie is reaped even after the machinery
/// under test kills the process.
struct StandIn {
    child: Child,
}

impl StandIn {
    fn spawn() -> Result<Self, Box<dyn Error>> {
        let child = Command::new("sleep").arg("300").process_group(0).spawn()?;
        Ok(Self { child })
    }

    fn pid(&self) -> u32 {
        self.child.id()
    }

    fn starttime(&self) -> Result<u64, Box<dyn Error>> {
        support::leader_starttime(self.pid()).ok_or_else(|| "missing leader start time".into())
    }
}

impl Drop for StandIn {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn dead_pid() -> Result<u32, Box<dyn Error>> {
    let mut child = Command::new("true").spawn()?;
    let pid = child.id();
    child.wait()?;
    Ok(pid)
}

#[test]
fn guard_reaps_registered_groups_on_drop() -> Result<(), Box<dyn Error>> {
    let registry = tempfile::tempdir()?;
    let workspace = tempfile::tempdir()?;
    let stand_in = StandIn::spawn()?;
    let guard =
        support::ServeReaper::with_registry(registry.path().to_path_buf(), workspace.path());
    let entry = support::register_group_in(
        registry.path(),
        stand_in.pid(),
        std::process::id(),
        stand_in.starttime()?,
        workspace.path(),
    )?;
    assert!(support::group_alive(stand_in.pid()));

    drop(guard);

    assert!(!support::group_alive(stand_in.pid()));
    assert!(!entry.exists());
    Ok(())
}

#[test]
fn guard_reaps_registered_groups_on_panic() -> Result<(), Box<dyn Error>> {
    let registry = tempfile::tempdir()?;
    let workspace = tempfile::tempdir()?;
    let stand_in = StandIn::spawn()?;
    let entry = support::register_group_in(
        registry.path(),
        stand_in.pid(),
        std::process::id(),
        stand_in.starttime()?,
        workspace.path(),
    )?;

    // The unwind under test needs a real panic.
    #[allow(clippy::panic)]
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        let _guard =
            support::ServeReaper::with_registry(registry.path().to_path_buf(), workspace.path());
        panic!("simulated test failure");
    }));

    assert!(outcome.is_err());
    assert!(!support::group_alive(stand_in.pid()));
    assert!(!entry.exists());
    Ok(())
}

#[test]
fn guard_spares_groups_of_other_workspaces() -> Result<(), Box<dyn Error>> {
    let registry = tempfile::tempdir()?;
    let workspace_a = tempfile::tempdir()?;
    let workspace_b = tempfile::tempdir()?;
    let stand_in_a = StandIn::spawn()?;
    let stand_in_b = StandIn::spawn()?;
    let guard =
        support::ServeReaper::with_registry(registry.path().to_path_buf(), workspace_a.path());
    support::register_group_in(
        registry.path(),
        stand_in_a.pid(),
        std::process::id(),
        stand_in_a.starttime()?,
        workspace_a.path(),
    )?;
    let entry_b = support::register_group_in(
        registry.path(),
        stand_in_b.pid(),
        std::process::id(),
        stand_in_b.starttime()?,
        workspace_b.path(),
    )?;

    drop(guard);

    assert!(!support::group_alive(stand_in_a.pid()));
    assert!(
        support::group_alive(stand_in_b.pid()),
        "a sibling test's still-serving fixture must survive another workspace's guard"
    );
    assert!(entry_b.exists());
    Ok(())
}

#[test]
fn sweep_reclaims_groups_with_dead_owners() -> Result<(), Box<dyn Error>> {
    let registry = tempfile::tempdir()?;
    let workspace = tempfile::tempdir()?;
    let stand_in = StandIn::spawn()?;
    let entry = support::register_group_in(
        registry.path(),
        stand_in.pid(),
        dead_pid()?,
        stand_in.starttime()?,
        workspace.path(),
    )?;

    support::sweep_orphaned_groups_in(registry.path());

    assert!(!support::group_alive(stand_in.pid()));
    assert!(!entry.exists());
    Ok(())
}

#[test]
fn sweep_skips_groups_with_live_owners() -> Result<(), Box<dyn Error>> {
    let registry = tempfile::tempdir()?;
    let workspace = tempfile::tempdir()?;
    let owned_by_self = StandIn::spawn()?;
    let owned_by_other = StandIn::spawn()?;
    let self_entry = support::register_group_in(
        registry.path(),
        owned_by_self.pid(),
        std::process::id(),
        owned_by_self.starttime()?,
        workspace.path(),
    )?;
    // The other stand-in doubles as a live foreign owner pid.
    let other_entry = support::register_group_in(
        registry.path(),
        owned_by_other.pid(),
        owned_by_other.pid(),
        owned_by_other.starttime()?,
        workspace.path(),
    )?;

    support::sweep_orphaned_groups_in(registry.path());

    assert!(support::group_alive(owned_by_self.pid()));
    assert!(support::group_alive(owned_by_other.pid()));
    assert!(self_entry.exists());
    assert!(other_entry.exists());
    Ok(())
}

#[test]
fn sweep_spares_recycled_group_identities() -> Result<(), Box<dyn Error>> {
    let registry = tempfile::tempdir()?;
    let workspace = tempfile::tempdir()?;
    let stand_in = StandIn::spawn()?;
    // A token that no longer matches the live leader models a pgid the
    // kernel recycled to an unrelated process after the recorded group died.
    let entry = support::register_group_in(
        registry.path(),
        stand_in.pid(),
        dead_pid()?,
        stand_in.starttime()? + 1,
        workspace.path(),
    )?;

    support::sweep_orphaned_groups_in(registry.path());

    assert!(
        support::group_alive(stand_in.pid()),
        "a mismatched identity token must never be signalled"
    );
    assert!(!entry.exists(), "the dead-owner entry itself is garbage");
    Ok(())
}

#[test]
fn port_lease_is_exclusive_while_owner_lives() -> Result<(), Box<dyn Error>> {
    let leases = tempfile::tempdir()?;
    assert!(support::try_lease_port_in(leases.path(), 45999));
    assert!(
        !support::try_lease_port_in(leases.path(), 45999),
        "a lease held by a live owner must not be reclaimed"
    );
    Ok(())
}

fn self_lease_content() -> Result<String, Box<dyn Error>> {
    let pid = std::process::id();
    let token = support::leader_starttime(pid).ok_or("missing own start time")?;
    Ok(format!("{pid}\n{token}"))
}

#[test]
fn port_lease_reclaims_dead_owners_only() -> Result<(), Box<dyn Error>> {
    let leases = tempfile::tempdir()?;
    fs::create_dir_all(leases.path())?;
    // Legacy pid-only lease with a dead owner.
    fs::write(leases.path().join("46001.lock"), dead_pid()?.to_string())?;
    assert!(
        support::try_lease_port_in(leases.path(), 46001),
        "a dead owner's legacy lease is reclaimable"
    );
    assert_eq!(
        fs::read_to_string(leases.path().join("46001.lock"))?,
        self_lease_content()?,
        "a fresh lease carries the owner pid and identity token"
    );

    fs::write(leases.path().join("46002.lock"), "mid-write garbage")?;
    assert!(
        !support::try_lease_port_in(leases.path(), 46002),
        "an unparseable lease is treated as held"
    );

    // A live pid with a mismatched token models a recycled owner pid: the
    // recorded owner is dead even though something now wears its pid.
    let recycled = StandIn::spawn()?;
    fs::write(
        leases.path().join("46003.lock"),
        format!("{}\n{}", recycled.pid(), recycled.starttime()? + 1),
    )?;
    assert!(
        support::try_lease_port_in(leases.path(), 46003),
        "a recycled owner pid does not keep the dead owner's lease held"
    );
    Ok(())
}

#[test]
fn lease_sweep_unlinks_exactly_the_stale_shapes() -> Result<(), Box<dyn Error>> {
    let leases = tempfile::tempdir()?;
    fs::create_dir_all(leases.path())?;
    let stand_in = StandIn::spawn()?;
    // Dead owner with its token; recycled pid (live process, wrong token);
    // legacy pid-only dead owner: all stale.
    fs::write(
        leases.path().join("47001.lock"),
        format!("{}\n1", dead_pid()?),
    )?;
    fs::write(
        leases.path().join("47002.lock"),
        format!("{}\n{}", stand_in.pid(), stand_in.starttime()? + 1),
    )?;
    fs::write(leases.path().join("47003.lock"), dead_pid()?.to_string())?;
    // Live owner and mid-write garbage: both held.
    fs::write(leases.path().join("47004.lock"), self_lease_content()?)?;
    fs::write(leases.path().join("47005.lock"), "mid-write garbage")?;

    support::sweep_stale_leases_in(leases.path());

    assert!(!leases.path().join("47001.lock").exists());
    assert!(!leases.path().join("47002.lock").exists());
    assert!(!leases.path().join("47003.lock").exists());
    assert!(
        leases.path().join("47004.lock").exists(),
        "a live owner's lease must survive the sweep"
    );
    assert!(
        leases.path().join("47005.lock").exists(),
        "an unparseable lease is held, not swept"
    );
    Ok(())
}

// The two interleave tests below are deterministically green on the locked
// implementation: the callee's first file access happens strictly after it
// acquires the directory lock, and the flip strictly precedes the release. A
// lock-free regression fails them only racily.

#[test]
fn sweep_defers_to_the_directory_lock() -> Result<(), Box<dyn Error>> {
    let leases = tempfile::tempdir()?;
    fs::create_dir_all(leases.path())?;
    let stale = leases.path().join("48001.lock");
    fs::write(&stale, dead_pid()?.to_string())?;
    let guard = support::lock_lease_dir(leases.path()).ok_or("failed to lock lease dir")?;
    let dir = leases.path().to_path_buf();
    let sweeper = std::thread::spawn(move || support::sweep_stale_leases_in(&dir));
    // Models a reclaiming binary taking the stale lease over while the sweep
    // still holds its pre-lock verdict.
    fs::write(&stale, self_lease_content()?)?;
    drop(guard);
    sweeper.join().map_err(|_| "sweep thread panicked")?;
    assert_eq!(
        fs::read_to_string(&stale)?,
        self_lease_content()?,
        "a lease that turned live before the sweep acquired the lock must survive"
    );
    Ok(())
}

#[test]
fn reclaim_rejudges_under_the_directory_lock() -> Result<(), Box<dyn Error>> {
    let leases = tempfile::tempdir()?;
    fs::create_dir_all(leases.path())?;
    let stale = leases.path().join("48002.lock");
    fs::write(&stale, dead_pid()?.to_string())?;
    let guard = support::lock_lease_dir(leases.path()).ok_or("failed to lock lease dir")?;
    let dir = leases.path().to_path_buf();
    let reclaimer = std::thread::spawn(move || support::try_lease_port_in(&dir, 48002));
    fs::write(&stale, self_lease_content()?)?;
    drop(guard);
    assert!(
        !reclaimer.join().map_err(|_| "reclaim thread panicked")?,
        "a lease that turned live in the window must be declined"
    );
    assert_eq!(fs::read_to_string(&stale)?, self_lease_content()?);
    Ok(())
}

#[test]
fn reclaim_creates_fresh_when_the_lease_vanished() -> Result<(), Box<dyn Error>> {
    let leases = tempfile::tempdir()?;
    let lock = leases.path().join("48003.lock");
    assert!(
        support::reclaim_collided_lease(leases.path(), &lock),
        "a lease swept away between the collision and the lock acquisition is free"
    );
    assert_eq!(fs::read_to_string(&lock)?, self_lease_content()?);
    Ok(())
}

#[test]
fn the_guard_file_is_never_judged_or_swept() -> Result<(), Box<dyn Error>> {
    let leases = tempfile::tempdir()?;
    fs::create_dir_all(leases.path())?;
    // Even wearing stale-looking lease content, the guard is not a lease.
    fs::write(leases.path().join("guard.flock"), dead_pid()?.to_string())?;
    support::sweep_stale_leases_in(leases.path());
    assert!(leases.path().join("guard.flock").exists());
    Ok(())
}

#[test]
fn claimed_ports_are_never_rehanded_in_process() {
    // Port 1 is never OS-assigned, so this cannot collide with a real
    // reservation sharing the binary-wide claim set.
    assert!(support::claim_port(1));
    assert!(!support::claim_port(1));
}

#[test]
fn reserved_ports_are_distinct_and_leased() -> Result<(), Box<dyn Error>> {
    let leases = tempfile::tempdir()?;
    let reserved = support::reserve_local_ports_in(leases.path(), 9)?;
    let mut ports = (0..9).map(|index| reserved.get(index)).collect::<Vec<_>>();
    ports.sort_unstable();
    ports.dedup();
    assert_eq!(ports.len(), 9, "reserved ports must be distinct");
    for port in &ports {
        assert_eq!(
            fs::read_to_string(leases.path().join(format!("{port}.lock")))?,
            self_lease_content()?,
            "every reserved port carries this process's lease and token"
        );
    }
    reserved.release();
    Ok(())
}

/// The registry file layout is a protocol between the Python shims and the
/// Rust reaper; this pins it end-to-end with the real shim artifact instead
/// of a transliterated copy.
#[test]
fn fixture_server_shim_registers_and_is_reaped() -> Result<(), Box<dyn Error>> {
    let registry = tempfile::tempdir()?;
    let workspace = tempfile::tempdir()?;
    let leases = tempfile::tempdir()?;
    let guard =
        support::ServeReaper::with_registry(registry.path().to_path_buf(), workspace.path());
    let shim = workspace.path().join("fixture-server");
    fs::write(&shim, include_str!("fixtures/bin/fixture-server.py"))?;
    let ports = support::reserve_local_ports_in(leases.path(), 1)?;
    let port = ports.get(0);
    ports.release();
    let child = Command::new("python3")
        .arg(&shim)
        .args(["127.0.0.1", &port.to_string()])
        .process_group(0)
        .envs(guard.env())
        .stdout(std::process::Stdio::null())
        .spawn()?;
    let mut stand_in = StandIn { child };

    let entry = registry.path().join(format!("{}.grp", stand_in.pid()));
    for _ in 0..50 {
        if entry.exists() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    assert!(entry.exists(), "the shim registers its group at startup");
    assert!(support::group_alive(stand_in.pid()));

    drop(guard);

    assert!(
        !support::group_alive(stand_in.pid()),
        "the guard parses the shim-written entry and reaps the group"
    );
    assert!(!entry.exists());
    let _ = stand_in.child.wait();
    Ok(())
}
