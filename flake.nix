{
  inputs = {
    nixpkgs = {
      url = "github:CrazyChaoz/nixpkgs/update-tflite-to-2.19";
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


        renamed_tflite = customPkgs: customPkgs.runCommand "renamed-tflite" {} 
          ''
            mkdir $out/
            mkdir $out/lib/

            # Copy everything
            cp -r ${customPkgs.tensorflow-lite}/* $out/

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
          ''
        ;

        buildMyRustThingy = customPkgs: (crane.mkLib customPkgs).buildPackage {
          src = ./.;
          doCheck = false;

          TFLITE_X86_64_LIB_DIR = "${(renamed_tflite customPkgs)}/lib";
          TFLITE_LIB_DIR = "${(renamed_tflite customPkgs)}/lib";

          buildInputs = with customPkgs; [
            (renamed_tflite customPkgs)
          ];
          nativeBuildInputs = with customPkgs; [
            clang
            pkg-config
            perl
            rustPlatform.bindgenHook
            cmake
            libclang
          ];
        };
      in
      {
        packages.default = buildMyRustThingy pkgs;

        packages.aarch64-linux = buildMyRustThingy pkgs.pkgsCross.aarch64-multiplatform;

        devShell = pkgs.mkShell {
          TFLITE_X86_64_LIB_DIR = "${renamed_tflite}/lib";
          TFLITE_LIB_DIR = "${renamed_tflite}/lib";

          buildInputs = with pkgs; [
            clang
          ];
          nativeBuildInputs = with pkgs; [
            rustc
            cargo
            renamed_tflite
            pkg-config
            clang
            libclang
            rustPlatform.bindgenHook
          ];
        };
      }
    );
}
