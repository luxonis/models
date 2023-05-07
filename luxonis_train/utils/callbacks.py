from pytorch_lightning.callbacks import RichProgressBar
from rich.table import Table
from rich.rule import Rule

class LuxonisProgressBar(RichProgressBar):
    """ Custom rich text progress bar based on RichProgressBar from Pytorch Lightning"""
    def __init__(self):
        # TODO: play with values to create custom output            
        # from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
        # progress_bar = RichProgressBar(
        #     theme = RichProgressBarTheme(
        #     description="green_yellow",
        #     progress_bar="green1",
        #     progress_bar_finished="green1",
        #     batch_progress="green_yellow",
        #     time="gray82",
        #     processing_speed="grey82",
        #     metrics="yellow1"
        #     )
        # )

        super().__init__(leave=True)

    def print_results(self, stage: str, loss: float, metrics: dict):
        """ Prints results to the console using rich text"""
        rule = Rule(stage, style="bold magenta")
        self._console.print(rule)
        self._console.print(f"[bold magenta]Loss:[/bold magenta] [white]{loss}[/white]")
        self._console.print(f"[bold magenta]Metrics:[/bold magenta]")
        for head in metrics:
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Metric name", style="magenta")
            table.add_column(head)
            for metric_name in metrics[head]:
                value = "{:.5f}".format(metrics[head][metric_name].item())
                table.add_row(metric_name, value)
            self._console.print(table)
        
        rule = Rule(style="bold magenta")
        self._console.print(rule)