use std::fs;
use std::io::{Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::UnixListener;
use std::path::PathBuf;

fn main() {
    let root = std::env::var_os("SYSTEM_X_CONTROL_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from("/home/user/SYSTEMS/system-x/INSPECTOR/RUNTIME/control-plane")
        });
    let _ = fs::create_dir_all(&root);
    let _ = fs::set_permissions(&root, fs::Permissions::from_mode(0o700));
    let socket = root.join("runtime-guard.sock");
    let _ = fs::remove_file(&socket);
    let listener = UnixListener::bind(&socket).expect("owner-only guard socket");
    fs::set_permissions(&socket, fs::Permissions::from_mode(0o600)).expect("guard socket mode");
    println!("system-x-runtime-guard private owner-only ready");
    for mut stream in listener.incoming().flatten() {
        let mut request = [0u8; 4096];
        if let Ok(size) = stream.read(&mut request) {
            if size <= 4096 && request[..size].windows(5).any(|w| w == b"READY") {
                let _ = stream.write_all(b"{\"status\":\"READY\",\"owner\":\"runtime-guard\"}\n");
            } else {
                let _ = stream.write_all(b"{\"status\":\"REJECTED\"}\n");
            }
        }
    }
}
