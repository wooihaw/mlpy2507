# mlpy2507

This repository contains the code and data for the Machine Learning with Python course.

## Setup

### 1. Install uv

`uv` is a fast Python package installer.

#### Windows

Open a PowerShell terminal and run the following command:

```shell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Alternatively, you can use the `winget` package manager if you have it installed:

```shell
winget install --id=astral-sh.uv -e
```

After installation, you may need to restart your PowerShell session for the `uv` command to be recognized.

#### Linux

On Linux, you can install `uv` using `curl` or `wget`. Open your terminal and execute one of the following commands:

Using `curl` (most common):

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Using `wget`:

```shell
wget -qO- https://astral.sh/uv/install.sh | sh
```

After running the installation script, you might need to restart your shell or source your profile file (e.g., `source ~/.bashrc` or `source ~/.zshrc`) for the `uv` command to become available in your `PATH`.

### 2. Clone the repository

Clone the repository to your local machine:

```shell
git clone https://github.com/wooihaw/mlpy2507.git
cd mlpy2507
```

### 3. Launch Jupyter Lab

Run the following command to launch Jupyter Lab:

```shell
uv run jupyter-lab
```
