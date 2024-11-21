{
  inputs = {
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    naersk = {
      url = "github:nix-community/naersk";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    nixpkgs.url = "nixpkgs/nixos-unstable";
    utils = {
      url = "github:numtide/flake-utils";
    };
  };

  outputs = { self, naersk, fenix, nixpkgs, utils }:
    utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        fixed-tensorflow-lite = (pkgs.tensorflow-lite).overrideAttrs (self: super: { meta.knownVulnerabilities = [ ]; });


        stdenv = pkgs.clangStdenv;

        toolchain = with fenix.packages.${system}; combine [
          minimal.cargo
          minimal.rustc
        ];
        pname = "tflite-rs";

        renamed_tflite = stdenv.mkDerivation {
          name = "renamed_tflite";
          src = fixed-tensorflow-lite;
          buildPhase = ''
            mkdir $out/
            mkdir $out/lib/

            # Copy everything except for specific files in lib/
            cp -r $src/* $out/

            # Rename specific files in lib/
            if [ -d $out/lib ]; then
              # Rename libtensorflowlite_c.so to libtensorflow-lite_c.so
              if [ -f $out/lib/libtensorflowlite_c.so ]; then
                cp $out/lib/libtensorflowlite_c.so $out/lib/libtensorflow-lite_c.so
              fi

              # Rename libtensorflowlite.so to libtensorflow-lite.so
              if [ -f $out/lib/libtensorflowlite.so ]; then
                cp $out/lib/libtensorflowlite.so $out/lib/libtensorflow-lite.so
              fi
            fi
          '';
        };

      in
      {
        devShell = pkgs.mkShell {
          TFLITE_X86_64_LIB_DIR = "${fixed-tensorflow-lite}/lib";
          TFLITE_LIB_DIR = "${fixed-tensorflow-lite}/lib";
          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
          buildInputs = with pkgs;[
            clang
            llvmPackages.libclang.lib
          ];
          nativeBuildInputs = with pkgs; [
            rustc
            cargo
            fixed-tensorflow-lite
            pkg-config
            opencv
            perl
            clang
            libclang
            vtk
            rustPlatform.bindgenHook
          ];
        };
      });
}
