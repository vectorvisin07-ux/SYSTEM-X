use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;

fn main() {
    let root = std::env::var_os("SYSTEM_X_CONTROL_ROOT").map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/home/user/SYSTEMS/system-x/INSPECTOR/RUNTIME/control-plane"));
    let _ = fs::create_dir_all(&root);
    let _ = fs::set_permissions(&root, fs::Permissions::from_mode(0o700));
    println!("system-x-runtime-guard private owner-only ready");
}
