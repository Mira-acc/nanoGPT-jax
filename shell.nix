let
  pkgs = import <nixpkgs> {};
in pkgs.mkShell {
  packages = with pkgs; [
    python3Full
    python3Packages.pip
    poetry
  ];

  # Tell Poetry which Python to use at runtime instead
  shellHook = ''
    export POETRY_PYTHON=${pkgs.python3}/bin/python
  '';
}
