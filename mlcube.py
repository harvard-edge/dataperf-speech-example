"""MLCube handler file"""
import os
import typer
import subprocess


app = typer.Typer()


class DownloadTask:
    """Download samples and eval data"""

    @staticmethod
    def run(parameters_file: str, output_path: str) -> None:

        cmd = "python3 utils/download_data.py"
        cmd += f" --parameters_file={parameters_file} --output_path={output_path}"
        splitted_cmd = cmd.split()

        process = subprocess.Popen(splitted_cmd, cwd=".")
        process.wait()


class SelectTask:
    """Execute selection algorithm"""

    @staticmethod
    def run(data_path: str, output_path: str) -> None:

        cmd = "python3 -m selection.main"
        cmd += f" --input_samples {data_path}/samples.pb --outdir={output_path}"
        splitted_cmd = cmd.split()

        process = subprocess.Popen(splitted_cmd, cwd=".")
        process.wait()


class EvaluateTask:
    """Execute evaluation script"""

    @staticmethod
    def run(eval_path: str, train_path: str, log_path: str) -> None:

        env = os.environ.copy()
        env.update({
            'eval_path': eval_path,
            'train_path': train_path,
            'log_path': log_path
        })

        process = subprocess.Popen("./utils/run_evaluate.sh", cwd=".", env=env)
        process.wait()


@app.command("download")
def download(
    parameters_file: str = typer.Option(..., "--parameters_file"),
    output_path: str = typer.Option(..., "--output_path"),
):
    DownloadTask.run(parameters_file, output_path)


@app.command("select")
def select(
    data_path: str = typer.Option(..., "--data_path"),
    output_path: str = typer.Option(..., "--output_path"),
):
    SelectTask.run(data_path, output_path)


@app.command("evaluate")
def evaluate(
    eval_path: str = typer.Option(..., "--eval_path"),
    train_path: str = typer.Option(..., "--train_path"),
    log_path: str = typer.Option(..., "--log_path"),
):
    EvaluateTask.run(eval_path, train_path, log_path)


if __name__ == "__main__":
    app()
