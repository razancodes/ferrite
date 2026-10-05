fn main() {
    // windows-gnu links WebView2Loader.dll dynamically: put it next to the exe
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("gnu") && std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
        // OUT_DIR = target/<profile>/build/<pkg>-<hash>/out
        if let Some(profile_dir) = out.ancestors().nth(3) {
            let src = std::path::Path::new("bin/WebView2Loader.dll");
            let _ = std::fs::copy(src, profile_dir.join("WebView2Loader.dll"));
        }
        println!("cargo:rerun-if-changed=bin/WebView2Loader.dll");
    }
    tauri_build::build()
}
