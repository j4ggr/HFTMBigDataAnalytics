{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  packages = with pkgs; [
    graphviz
    uv
  ];

  shellHook = ''
    export UV_PROJECT_ENVIRONMENT="$PWD/.venv"

    if [ ! -d "$UV_PROJECT_ENVIRONMENT" ]; then
      uv venv "$UV_PROJECT_ENVIRONMENT"
    fi

    uv pip sync requirements.txt
    source "$UV_PROJECT_ENVIRONMENT/bin/activate"

    echo "Run the Streamlit app with: uv run streamlit run streamlit_app.py"
  '';
}
