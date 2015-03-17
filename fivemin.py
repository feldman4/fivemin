import csv
import re
import pandas as pd
import numpy as np

test_form = 'reaction_setup2.csv'
output_csv = 'output2.csv'
concentration_pattern = '([0-9]*\.*[0-9]*)(.*)'
named_series_pattern = '((.*):)*(.*)'
reaction_volume = 10
pipette_loss = 1.1


class Experiment(object):
    def __init__(self, form, reaction_volume=10, pipette_loss=1.):
        """Container for experiment, including all parameters and generated information.
        :param form:
        :return:
        """
        self.info = {}
        self.split_volume = {}
        self.total_volume = {}
        self.components = []
        self.splits = {}
        self.reactions = {}
        self.volumes = {}
        self.reaction_volume = reaction_volume
        self.pipette_loss = pipette_loss

        self.setup(form)


    def setup(self, form):
        """Extract Components and desired conditions from pandas DataFrame.
        :param form:
        :return:
        """
        for rank, name, stock in zip(form.fillna(value=0)['rank'], form['reagent'], form['stock']):
            self.components.append(Component(name=name, rank=rank, stock=Concentration(stock)))

        experiments = filter(lambda s: 'experiment' in s, form.columns)

        for experiment in experiments:
            # TODO convert concentrations to fold dilutions
            index, indices, named_index = 0, {}, {}
            # singletons will be given index 0
            split = {0: []}
            for entry, component in zip(form.fillna(value='1X')[experiment], self.components):
                # generic format A: 1, 2, 3
                name, series = re.match(named_series_pattern, entry).groups()[1:3]
                ingredient = [component, [Concentration(c.strip()) for c in series.split(',')]]
                # assign unique index to each step in the reaction, sort later
                if name is not None:
                    if name in named_index:
                        # previously recognized series
                        split[named_index[name]].append(ingredient)
                    else:
                        # new series
                        index += 1
                        named_index[name] = index
                        split[index] = [ingredient]
                elif len(ingredient[1]) == 1:
                    # singleton
                    split[0].append(ingredient)
                else:
                    # unnamed series, gets its own split
                    index += 1
                    split[index] = [ingredient]

            split = self.sort_splits([split[i] for i in range(index + 1)])

            # calculate volumes
            split_sizes = [len(s[0][1]) for s in split]
            total_volume = np.array([self.pipette_loss ** i *
                                     self.reaction_volume * np.prod(split_sizes)
                                     for i in reversed(range(len(split)))])
            split_volume = np.array(total_volume) / np.cumprod(split_sizes)
            total_volume = total_volume

            for ingredients, vol in zip(split, split_volume):
                for ing in ingredients:
                    stock_c = ing[0].stock.value
                    ing.append(np.around([vol * (c.value / stock_c) for c in ing[1]], decimals=2))
            self.info[experiment] = {'split_volume': split_volume,
                                     'total_volume': total_volume,
                                     'split_sizes': split_sizes}
            self.splits[experiment] = split
            self.reactions[experiment] = self.expand(split)

    def sort_splits(self, split):
        # permute, sorting by lowest rank of split then size

        rank = [max([i[0].rank for i in ingredients]) for ingredients in split]
        rank = [len(ingredients[0][1]) if r == 0 else 1000 + r for r, ingredients in zip(rank, split)]
        sorted_split = [s for (r, s) in sorted(zip(rank, split))]
        return sorted_split

    def expand(self, split):
        """Represent experiment layout as ndarray of Reactions with order of axes corresponding to
        split order. Associate each Component with an axis.
        :return:
        """
        shape = [len(ingredients[0][1]) for ingredients in split]
        reactions = np.ndarray(shape, dtype=Reaction)
        it = np.nditer(reactions, op_flags=['readwrite'], flags=['f_index', 'multi_index', 'refs_ok'])
        for r in it:
            r[...] = Reaction()
            for i, split_index in enumerate(it.multi_index):
                ingredients = split[i]
                for ingredient in ingredients:
                    reactions.ravel(order='F')[it.index].add_by_concentration(ingredient[0], ingredient[1][split_index])
        return reactions

    def layout(self):
        layout = []
        for experiment, reactions in self.reactions.items():
            layout.append([experiment])
            block_shape = reactions.shape[-2:]
            if len(block_shape) == 1:
                block_shape = (1, block_shape[0])
            new_shape = (reactions.size / block_shape[1], block_shape[1])
            for line in reactions.reshape(new_shape).tolist():
                layout.append(line)
        return layout


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
        return "%.2g %s" % (self.value, self.units)


class Reaction(object):
    def __init__(self, volume=1):
        self.concentrations = {}
        self.volume = float(volume)

    def add_by_concentration(self, component, concentration):
        """If Component already in Reaction, overwrites pre-existing Concentration.
        :param component:
        :param concentration:
        :return:
        """
        self.concentrations[component] = concentration

    def __repr__(self):
        representation = []
        for component, concentration in self.concentrations.items():
            representation.append(str(component) + ': ' + str(concentration))
        return "{%s}" % '; '.join(representation)


class Component(object):
    def __init__(self, name, rank, stock):
        self.name = name
        self.rank = rank
        self.stock = stock

    def __repr__(self):
        return "%s" % self.name


def test():
    form = pd.read_table(test_form, sep=',')
    exp = Experiment(form, reaction_volume=reaction_volume, pipette_loss=1.1)
    csv.writer(open(output_csv, 'wb')).writerows(exp.layout())
    return exp


if __name__ == '__main__':
    test()