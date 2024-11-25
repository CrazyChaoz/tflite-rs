{
  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    utils = {
      url = "github:numtide/flake-utils";
    };

    crane.url = "github:ipetkov/crane";
  };

  outputs = { self, nixpkgs, utils, crane }:
    utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        packages.default = (crane.mkLib pkgs).buildPackage {
          src = ./.;
          strictDeps = true;
          doCheck = true;

          TFLITE_X86_64_LIB_DIR = "${pkgs.tensorflow-lite}/lib";
          TFLITE_LIB_DIR = "${pkgs.tensorflow-lite}/lib";

          buildInputs = with pkgs; [
            tensorflow-lite
            vtk
          ];
          nativeBuildInputs = with pkgs; [
            clang
            pkg-config
            perl
            rustPlatform.bindgenHook
            cmake
            libclang
          ];
        };

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
