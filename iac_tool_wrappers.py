import subprocess
import json
from langchain_core.tools import tool

# --- Configuration ---
# NOTE: In a real system, this directory must be a clean, isolated volume
# e.g., a temporary directory created for each run.
IAC_TEMP_DIR = "./temp_iac_repo"
CODE_FILENAME = "main.tf"

def _run_terraform_command(command: list[str]) -> tuple[int, str]:
    """Helper to run a Terraform command and capture output."""
    try:
        # NOTE: Ensure 'terraform' is in the PATH of the execution environment (Docker)
        result = subprocess.run(
            command,
            cwd=IAC_TEMP_DIR,
            capture_output=True,
            text=True,
            check=False # Do not raise exception on non-zero exit code
        )
        return result.returncode, result.stdout + result.stderr
    except FileNotFoundError:
        return 1, "Error: Terraform CLI not found in execution path."
    except Exception as e:
        return 1, f"Execution failed: {e}"

@tool
def validate_terraform_code(code: str) -> str:
    """
    Writes the provided Terraform code to a file, runs 'terraform init' and 
    'terraform validate', and returns a structured JSON result indicating success 
    or detailed error messages for self-correction.
    
    Args:
        code: The full Terraform HCL code block (string).
        
    Returns:
        A JSON string containing the validation result.
    """
    # 1. Setup Environment
    subprocess.run(["mkdir", "-p", IAC_TEMP_DIR], check=True)
    
    # 2. Write Code to File
    with open(f"{IAC_TEMP_DIR}/{CODE_FILENAME}", "w") as f:
        f.write(code)

    # 3. Run 'terraform init'
    init_code, init_output = _run_terraform_command(["terraform", "init", "-no-color"])
    if init_code != 0:
        return json.dumps({
            "status": "FAIL",
            "reason": "INIT_ERROR",
            "details": f"Terraform Init Failed: Check provider/module blocks.\nOutput: {init_output}"
        })

    # 4. Run 'terraform validate'
    validate_code, validate_output = _run_terraform_command(["terraform", "validate", "-json"])
    
    # 5. Process Output
    if validate_code == 0:
        return json.dumps({
            "status": "PASS",
            "reason": "Validation successful."
        })
    else:
        # Attempt to parse structured JSON output from `terraform validate -json`
        try:
            # Terraform validate -json often emits HCL errors as a JSON object
            validate_json = json.loads(validate_output)
            # LLM needs specific error message to fix code
            error_message = validate_json.get("error_count", 0)
            return json.dumps({
                "status": "FAIL",
                "reason": "VALIDATION_ERROR",
                "details": validate_json.get("diagnostics", validate_output)
            })
        except json.JSONDecodeError:
            # Fallback to raw output if JSON parsing fails (e.g., initial syntax error)
            return json.dumps({
                "status": "FAIL",
                "reason": "VALIDATION_ERROR_RAW",
                "details": validate_output
            })