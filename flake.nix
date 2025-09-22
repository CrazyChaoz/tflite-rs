{
  inputs = {
    nixpkgs = {
      type = "git";
      url = "https://github.com/crazychaoz/nixpkgs.git";
      ref = "update-tflite-to-2.20";
    };

    utils.url = "github:numtide/flake-utils";
    crane.url = "github:ipetkov/crane";
  };
  outputs =
    {
      self,
      nixpkgs,
      utils,
      crane,
    }:
    utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };

        renamed_tflite = pkgs.clangStdenv.mkDerivation {
          name = "renamed_tflite";
          src = pkgs.tensorflow-lite;
          buildPhase = ''
            mkdir $out/
            mkdir $out/lib/

            # Copy everything
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
        packages.default = (crane.mkLib pkgs).buildPackage {
          src = ./.;
          doCheck = false;

          TFLITE_X86_64_LIB_DIR = "${renamed_tflite}/lib";
          TFLITE_LIB_DIR = "${renamed_tflite}/lib";

          buildInputs = [
            pkgs.renamed_tflite
          ];
          nativeBuildInputs = with pkgs; [
            clang
            pkg-config
            perl
            rustPlatform.bindgenHook
            cmake
            libclang
            rustfmt
          ];
        };

        devShell = pkgs.mkShell {
          TFLITE_X86_64_LIB_DIR = "${renamed_tflite}/lib";
          TFLITE_LIB_DIR = "${renamed_tflite}/lib";

          buildInputs = [
            pkgs.renamed_tflite
          ];
          nativeBuildInputs = with pkgs; [
            rustc
            cargo
            renamed_tflite
            pkg-config
            clang
            libclang
            rustPlatform.bindgenHook
            rustfmt
          ];
        };
      }
    );
}
