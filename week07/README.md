# PyTorch Template Tutorial

## Prerequisites

- Install [uv](https://docs.astral.sh/uv/)
  ```bash
  # Linux or macOS
  curl -LsSf https://astral.sh/uv/install.sh | sh

  # Windows
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

## Use template

1. Go to [pytorch_template](https://github.com/Axect/pytorch_template)

2. Click on the green "Use this template" button (upper right corner)

3. Select "Create a new repository"

4. Fill in the repository name and description (e.g. `qu4_template_tutorial`)

5. Select the visibility (public or private)

6. Click on the green "Create repository from template" button

7. Clone the repository to your local machine
   ```bash
   git clone https://github.com/<your_username>/qu4_template_tutorial
   ```

8. Change directory to the cloned repository
   ```bash
   cd qu4_template_tutorial
   ```

9. Install the required packages
   ```bash
   # 1. Create a virtual environment
   uv venv

   # 2. Sync the requirements
   uv pip sync requirements.txt

   # 2. Or fresh install
   uv pip install -U torch wandb rich beaupy polars numpy optuna matplotlib scienceplots

   # 1. Or use pip (not recommended)
   pip install -r requirements.txt
   ```

10. Activate the virtual environment
    ```bash
    # Linux or macOS
    source .venv/bin/activate

    # Windows
    .venv\Scripts\activate
    ```
