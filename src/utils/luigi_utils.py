import luigi

from config import DEBUG
from src.utils.utils import PROJECT_DIR
from src.utils.utils import loop_through_iterable
from src.utils.utils import read_data, save_data

CACHE_DIR = PROJECT_DIR / 'cache' if not DEBUG else PROJECT_DIR / 'debug_cache'


class TaskWrpper(luigi.Task):
    DEBUG = luigi.BoolParameter(default=DEBUG)

    def input(self):
        if self.requires() is None:
            return None
        return super().input()

    def load_inputs(self):
        if self.input() is None:
            return []
        return [read_data(task.path) for task in self.input()]

    def load_output(self):
        if self.output() is None:
            return None
        return loop_through_iterable(self.output(), lambda o: read_data(o.path))

    def save_output(self, output):
        save_data(output, self.output().path)

    def build(self, local_scheduler=True, *args, **kwargs):
        luigi.build([
            self
        ],
            local_scheduler=local_scheduler, *args, **kwargs)
        return self.load_output()


def luigi_task(inputs, no_output=False, sample_frac=0.2):
    if inputs is not None and type(inputs) != list:
        inputs = [inputs]

    def luigi_task_dec(func):
        task_name = func.__name__

        class FuncTask(TaskWrpper):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._func = func

            def requires(self):
                return inputs

            def output(self):
                if not no_output:
                    return luigi.LocalTarget(path=CACHE_DIR / f'{task_name}.pickle')

            def run_func(self):
                output = self._func(*self.load_inputs())
                if self.input() is None and self.DEBUG:
                    output = output.sample(frac=sample_frac)
                return output

            def test(self):
                output = self.run_func()
                output_test = self.load_output()
                return output, output_test

            def run(self):
                output = self.run_func()
                if output is not None:
                    self.save_output(output)
                    return output

            def __call__(self, *args, **kwargs):
                return self._func(*args, **kwargs)

        FuncTask.__name__ = task_name if not DEBUG else 'debug_' + task_name
        FuncTask.__doc__ = func.__doc__

        return FuncTask()

    return luigi_task_dec

# class Pipeline(TaskWrpper):
#     def __init__(self, final_tasks):
#         super().__init__()
#         if type(final_tasks) != list:
#             final_tasks = [final_tasks]
#         self.final_tasks = final_tasks
#
#     def requires(self):
#         return self.final_tasks
