# -*- coding: utf-8 -*-
"""
Batchrunner
===========

A single class to manage a batch run or parameter sweep of a given model.

"""
import copy
from itertools import product, count
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import os

import random

class ParameterError(TypeError):
    MESSAGE = (
        "parameters must map a name to a value. "
        "These names did not match paramerets: {}"
    )

    def __init__(self, bad_names):
        self.bad_names = bad_names

    def __str__(self):
        return self.MESSAGE.format(self.bad_names)


class VariableParameterError(ParameterError):
    MESSAGE = (
        "variable_parameters must map a name to a sequence of values. "
        "These parameters were given with non-sequence values: {}"
    )

    def __init__(self, bad_names):
        super().__init__(bad_names)


class FixedBatchRunner:
    """ This class is instantiated with a model class, and model parameters
    associated with one or more values. It is also instantiated with model and
    agent-level reporters, dictionaries mapping a variable name to a function
    which collects some data from the model or its agents at the end of the run
    and stores it.

    Note that by default, the reporters only collect data at the *end* of the
    run. To get step by step data, simply have a reporter store the model's
    entire DataCollector object.
    """

    def __init__(
        self,
        model_cls,
        parameters_list=None,
        fixed_parameters=None,
        iterations=1,
        max_steps=1000,
        model_reporters=None,
        agent_reporters=None,
        display_progress=True,
    ):
        """ Create a new BatchRunner for a given model with the given
        parameters.

        Args:
            model_cls: The class of model to batch-run.
            parameters_list: A list of dictionaries of parameter sets.
                The model will be run with dictionary of paramters.
                For example, given parameters_list of
                    [{"homophily": 3, "density": 0.8, "minority_pc": 0.2},
                    {"homophily": 2, "density": 0.9, "minority_pc": 0.1},
                    {"homophily": 4, "density": 0.6, "minority_pc": 0.5}]
                3 models will be run, one for each provided set of parameters.
            fixed_parameters: Dictionary of parameters that stay same through
                all batch runs. For example, given fixed_parameters of
                    {"constant_parameter": 3},
                every instantiated model will be passed constant_parameter=3
                as a kwarg.
            iterations: The total number of times to run the model for each set
                of parameters.
            max_steps: Upper limit of steps above which each run will be halted
                if it hasn't halted on its own.
            model_reporters: The dictionary of variables to collect on each run
                at the end, with variable names mapped to a function to collect
                them. For example:
                    {"agent_count": lambda m: m.schedule.get_agent_count()}
            agent_reporters: Like model_reporters, but each variable is now
                collected at the level of each agent present in the model at
                the end of the run.
            display_progress: Display progresss bar with time estimation?

        """
        self.model_cls = model_cls

        if parameters_list is None:
            parameters_list = []
        self.parameters_list = list(parameters_list)
        self.fixed_parameters = fixed_parameters or {}
        self._include_fixed = len(self.fixed_parameters.keys()) > 0
        self.iterations = iterations
        self.max_steps = max_steps

        self.model_reporters = model_reporters
        self.agent_reporters = agent_reporters

        if self.model_reporters:
            self.model_vars = {}

        if self.agent_reporters:
            self.agent_vars = {}

        self.display_progress = display_progress

    def _make_model_args(self):
        """Prepare all combinations of parameter values for `run_all`

        Returns:
            Tuple with the form:
            (total_iterations, all_kwargs, all_param_values)
        """
        total_iterations = self.iterations
        all_kwargs = []

        count = len(self.parameters_list)
        if count:
            for params in self.parameters_list:
                kwargs = params.copy()
                kwargs.update(self.fixed_parameters)
                #run each iterations specific number of times
                for iter in range(self.iterations):
                    kwargs_repeated = kwargs.copy()
                    all_kwargs.append([self.model_cls, kwargs_repeated, self.max_steps, iter])

        elif len(self.fixed_parameters):
            count = 1
            kwargs = self.fixed_parameters.copy()
            all_kwargs.append(kwargs)

        total_iterations *= count

        return all_kwargs, total_iterations
        #return (total_iterations, all_kwargs, all_param_values)

    def run_all(self):
        """ Run the model at all parameter combinations and store results. """
        run_count = count()
        total_iterations, all_kwargs, all_param_values = self._make_model_args()

        with tqdm(total_iterations, disable=not self.display_progress) as pbar:
            for i, kwargs in enumerate(all_kwargs):
                param_values = all_param_values[i]
                for _ in range(self.iterations):
                    self.run_iteration(kwargs, param_values, next(run_count))
                    pbar.update()

    @staticmethod
    def run_wrapper(iter_args):
        model_i = iter_args[0]
        kwargs = iter_args[1]
        max_steps = iter_args[2]
        iteration = iter_args[3]

        def run_iteration(model_i, kwargs, max_steps, iteration):
            #instantiate version of model with correct parameters
            model = model_i(**kwargs)
            while model.running and model.schedule.steps < max_steps:
                model.step()

            #add iteration number to dictionary to make unique_key
            kwargs["iteration"] = iteration


            if model.datacollector:
                return kwargs, model.datacollector.get_model_vars_dataframe()
            else:
                return kwargs, "no datacollector in model"

        return run_iteration(model_i, kwargs, max_steps, iteration)


    def run_model(self, model):
        """ Run a model object to completion, or until reaching max steps.

        If your model runs in a non-standard way, this is the method to modify
        in your subclass.

        """
        while model.running and model.schedule.steps < self.max_steps:
            model.step()
            count+=1


    def collect_model_vars(self, model):
        """ Run reporters and collect model-level variables. """
        model_vars = {}
        for var, reporter in self.model_reporters.items():
            model_vars[var] = reporter(model)
        return model_vars

    def collect_agent_vars(self, model):
        """ Run reporters and collect agent-level variables. """
        agent_vars = {}
        for agent in model.schedule._agents.values():
            agent_record = {}
            for var, reporter in self.agent_reporters.items():
                agent_record[var] = getattr(agent, reporter)
            agent_vars[agent.unique_id] = agent_record
        return agent_vars


    def get_model_vars_dataframe(self):
        """ Generate a pandas DataFrame from the model-level variables
        collected.

        """
        return self._prepare_report_table(self.model_vars)

    def get_agent_vars_dataframe(self):
        """ Generate a pandas DataFrame from the agent-level variables
        collected.

        """
        return self._prepare_report_table(self.agent_vars, extra_cols=["AgentId"])

    def _prepare_report_table(self, vars_dict, extra_cols=None):
        """
        Creates a dataframe from collected records and sorts it using 'Run'
        column as a key.
        """
        extra_cols = ["Run"] + (extra_cols or [])
        index_cols = set()
        for params in self.parameters_list:
            index_cols |= params.keys()
        index_cols = list(index_cols) + extra_cols

        records = []
        for param_key, values in vars_dict.items():
            record = dict(zip(index_cols, param_key))
            record.update(values)
            records.append(record)

        df = pd.DataFrame(records)
        rest_cols = set(df.columns) - set(index_cols)
        ordered = df[index_cols + list(sorted(rest_cols))]
        ordered.sort_values(by="Run", inplace=True)
        if self._include_fixed:
            for param in self.fixed_parameters.keys():
                val = self.fixed_parameters[param]

                # avoid error when val is an iterable
                vallist = [val for i in range(ordered.shape[0])]
                ordered[param] = vallist
        return ordered


# This is kind of a useless class, but it does carry the 'source' parameters with it
class ParameterProduct:
    def __init__(self, variable_parameters):
        if variable_parameters != None:
            self.param_names, self.param_lists = zip(
                *(copy.deepcopy(variable_parameters)).items()
            )
            self._product = product(*self.param_lists)
        else:
            self.param_names =  None
            self.param_lists = None


    def __iter__(self):
        return self


    def __next__(self):
        if self.param_names != None:
            return dict(zip(self.param_names, next(self._product)))
        else:
            return []

# Roughly inspired by sklearn.model_selection.ParameterSampler.  Does not handle
# distributions, only lists.
class ParameterSampler:
    def __init__(self, parameter_lists, n, random_state=None):
        self.param_names, self.param_lists = zip(
            *(copy.deepcopy(parameter_lists)).items()
        )
        self.n = n
        if random_state is None:
            self.random_state = random.Random()
        elif isinstance(random_state, int):
            self.random_state = random.Random(random_state)
        else:
            self.random_state = random_state
        self.count = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.count += 1
        if self.count <= self.n:
            return dict(
                zip(
                    self.param_names,
                    [self.random_state.choice(l) for l in self.param_lists],
                )
            )
        raise StopIteration()


class BatchRunner(FixedBatchRunner):
    """ This class is instantiated with a model class, and model parameters
    associated with one or more values. It is also instantiated with model and
    agent-level reporters, dictionaries mapping a variable name to a function
    which collects some data from the model or its agents at the end of the run
    and stores it.

    Note that by default, the reporters only collect data at the *end* of the
    run. To get step by step data, simply have a reporter store the model's
    entire DataCollector object.

    """

    def __init__(
        self,
        model_cls,
        variable_parameters=None,
        fixed_parameters=None,
        iterations=1,
        max_steps=1000,
        model_reporters=None,
        agent_reporters=None,
        display_progress=True,
    ):
        """ Create a new BatchRunner for a given model with the given
        parameters.

        Args:
            model_cls: The class of model to batch-run.
            variable_parameters: Dictionary of parameters to lists of values.
                The model will be run with every combo of these paramters.
                For example, given variable_parameters of
                    {"param_1": range(5),
                     "param_2": [1, 5, 10]}
                models will be run with {param_1=1, param_2=1},
                    {param_1=2, param_2=1}, ..., {param_1=4, param_2=10}.
            fixed_parameters: Dictionary of parameters that stay same through
                all batch runs. For example, given fixed_parameters of
                    {"constant_parameter": 3},
                every instantiated model will be passed constant_parameter=3
                as a kwarg.
            iterations: The total number of times to run the model for each
                combination of parameters.
            max_steps: Upper limit of steps above which each run will be halted
                if it hasn't halted on its own.
            model_reporters: The dictionary of variables to collect on each run
                at the end, with variable names mapped to a function to collect
                them. For example:
                    {"agent_count": lambda m: m.schedule.get_agent_count()}
            agent_reporters: Like model_reporters, but each variable is now
                collected at the level of each agent present in the model at
                the end of the run.
            display_progress: Display progresss bar with time estimation?

        """
        super().__init__(
            model_cls,
            ParameterProduct(variable_parameters),
            fixed_parameters,
            iterations,
            max_steps,
            model_reporters,
            agent_reporters,
            display_progress,
        )


class MPSupport(Exception):
    def __str__(self):
        return (
            "BatchRunnerMP depends on pathos, which is either not "
            "installed, or the path can not be found. "
        )


class BatchRunnerMP(BatchRunner):
    """ Child class of BatchRunner, extended with multiprocessing support. """

    def __init__(self, model_cls, nr_processes=None, **kwargs):
        """ Create a new BatchRunnerMP for a given model with the given
        parameters.

        Args:
            model_cls: The class of model to batch-run.
            nr_processes: the number of separate processes the BatchRunner
                should start, all running in parallel.
            kwargs: the kwargs required for the parent BatchRunner class
        """
        if nr_processes == None:
            #identifies the number of processors available on users machine
            available_processors = cpu_count()
            self.processes = available_processors
            print ("Your system has {} available processors.".format(self.processes))
        else:
            self.processes = nr_processes

        super().__init__(model_cls, **kwargs)
        self.pool = Pool(self.processes)

    def run_all(self):
        """
        Run the model at all parameter combinations and store results,
        overrides run_all from BatchRunner.
        """
        run_count = count()
        run_iter_args, total_iterations = self._make_model_args()
        # register the process pool and init a queue
        #results = []
        results = {}
        #with tqdm(total_iterations, disable=not self.display_progress) as pbar:
            #for i, kwargs in enumerate(all_kwargs):
            #    param_values = all_param_values[i]
            #    for _ in range(self.iterations):
                    # make a new process and add it to the queue
            #with self.pool as p:
        if self.processes > 1:
            for params, model_data in self.pool.imap_unordered(self.run_wrapper, run_iter_args):
                results[str(params)] = model_data
        #For debugging model due to difficulty of getting errors during multiprocessing
        else:
            for run in run_iter_args:
                params, model_data = self.run_wrapper(run)
                #params, model_data = self.run_wrapper(run)
                #no need for a dictionary since one set of results
                results[str(params)] = model_data

        return results

        # empty the queue


        results = []
        for task in job_queue:
            for model_vars, agent_vars in list(task):
                results.append((model_vars, agent_vars))
            pbar.update()

        # store the results
        for model_vars, agent_vars in results:
            if self.model_reporters:
                for model_key, model_val in model_vars.items():
                    self.model_vars[model_key] = model_val
            if self.agent_reporters:
                for agent_key, reports in agent_vars.items():
                    self.agent_vars[agent_key] = reports


        with tqdm(total_iterations, disable=not self.display_progress) as pbar:
            for i, kwargs in enumerate(all_kwargs):
                param_values = all_param_values[i]
                for _ in range(self.iterations):
                    # make a list of parameters for each model run
                    job_queue.append((kwargs, param_values, next(run_count)))
                    #start dictionary to store results
                    #results[next(run_count)] =[param_values]

            # empty the queue
            results = []
            print (len(job_queue))
            with self.pool as p:
                results.append(p.imap_unordered(self.run_iteration, job_queue))
                pbar.update()