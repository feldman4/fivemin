import re
import pandas as pd

test_form = 'reaction_setup.csv'
concentration_pattern = '([0-9]*\.*[0-9]*)(.*)'


class Experiment(object):
    def __init__(self, form):
        """Container for experiment, including all parameters and generated information.
        :param form:
        :return:
        """
        self.components = []
        self.conditions = {}
        self.setup(form)
        self.expand()

    def setup(self, form):
        """Extract Components and desired conditions from pandas DataFrame.
        :param form:
        :return:
        """
        for rank, name, stock in zip(form.fillna(value=0)['rank'], form['reagent'], form['stock']):
            self.components.append(Component(name=name, rank=rank, stock=Concentration(stock)))

        experiments = filter(lambda s: 'experiment' in s, form.columns)

        for experiment in experiments:
            # convert concentrations to fold dilutions
            conditions = []
            #TODO allow for "parallel" concentration ranges (A: 1, 2, 3 and A: 1, 1, 2)
            for entry in form.fillna(value='1X')[experiment]:
                conditions.append([Concentration(c.strip()) for c in entry.split(',')])
            self.conditions[experiment] = conditions

    def expand(self):
        for conditions in self.conditions:
            #TODO recursivley identify next component, applying rank
            # find singletons
            [conditions.pop(component) for component in condition if len(component) == 1]


class Concentration(object):
    def __init__(self, input_string):
        self.units = None
        self.value = 0.
        self.parse(input_string)

    def parse(self, input_string):
        self.value, self.units = re.match(concentration_pattern, input_string).groups()[:2]
        self.value = float(self.value)
        self.units = self.units.strip()

    def __repr__(self):
        return "%d %s" % (self.value, self.units)


class Reaction(object):
    def __init__(self, components=None, volume=1):
        self.components = [] if components is None else components
        self.volume = float(volume)


class Component(object):
    def __init__(self, name, rank, stock):
        self.name = name
        self.rank = rank
        self.stock = stock.value


if __name__ == '__main__':
    form = pd.read_table(test_form, sep=',')
    exp = Experiment(form)
    print exp