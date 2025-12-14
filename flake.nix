{
  description = "champs-magnetiques";

  inputs.nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      forAllSystems = nixpkgs.lib.genAttrs nixpkgs.lib.systems.flakeExposed;
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = import nixpkgs { inherit system; };
        in
        {
          default = pkgs.mkShell {
            packages = with pkgs; [
              (python313.withPackages (
                python-pkgs: with python-pkgs; [
                  numpy
                  matplotlib
                  scipy
                  numba
                  pytest
                ]
              ))
              jetbrains.pycharm-community
            ];
          };
        }
      );
    };
}
