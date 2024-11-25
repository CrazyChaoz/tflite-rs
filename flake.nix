{
  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    utils = {
      url = "github:numtide/flake-utils";
    };
  };

  outputs = { self, nixpkgs, utils }:
    utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        devShell = pkgs.mkShell {
          TFLITE_X86_64_LIB_DIR = "${pkgs.tensorflow-lite}/lib";
          TFLITE_LIB_DIR = "${pkgs.tensorflow-lite}/lib";
          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
          buildInputs = with pkgs;[
            clang
          ];
          nativeBuildInputs = with pkgs; [
            rustc
            cargo
            tensorflow-lite
            pkg-config
            clang
            libclang
            rustPlatform.bindgenHook
          ];
        };
      });
}
