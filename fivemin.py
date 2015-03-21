import csv, string
import re
import pandas as pd
from collections import defaultdict
import numpy as np
import sympy as sp
from numpy.linalg import lstsq

test_form = 'reaction_tests.csv'
output_csv = 'output2.csv'
concentration_pattern = '([0-9]*\.*[0-9]*)(.*)'
named_series_pattern = '((.*):)*(.*)'
test_reaction_volume = 10
test_pipette_loss = 1.1


def symbol_array(sym, size, assumptions=None):
    return [sp.symbols(sym + '_' + str(i), **assumptions)
            for i in range(size)]


class Experiment(object):
    def __init__(self, form, reaction_volume=10, pipette_loss=1.):
        """Container for experiment, including all parameters and generated information.
        :param form:
        :return:
        """
        self.info = {}
        self.components = []
        self.reactions = {}
        self.reaction_volume = reaction_volume
        self.pipette_loss = pipette_loss
        self.syms = {}
        self.syms_to_components = {}
        self.experiments = {}
        self.expressions = {}
        self.uc_alphabet = (chr(i) for i in range(65, 65 + 26))

        self.setup(form)

    def setup(self, form):
        """Extract Components and desired conditions from pandas DataFrame.
        :param form:
        :return:
        """
        # unpack form
        for rank, name, stock in zip(form.fillna(value=0)['rank'], form['reagent'], form['stock']):
            self.components.append(Component(name=name, rank=rank, concentration=Concentration(stock)))
            self.syms[self.components[-1]] = self.uc_alphabet.next()
        self.syms_to_components = {val: key for key, val in self.syms.items()}
        # get experiment columns
        experiments = filter(lambda s: 'experiment' in s, form.columns)
        # determine sub-mixes and components
        self.experiments = {exp: defaultdict(list) for exp in experiments}
        for experiment in experiments:
            # TODO convert concentrations to fold dilutions
            for entry, component in zip(form.fillna(value='1X')[experiment], self.components):
                # generic format A: 1, 2, 3
                name, series = re.match(named_series_pattern, entry).groups()[1:3]
                series = [Concentration(c.strip(), stock=component) for c in series.split(',')]
                series = sorted(series, key=lambda s: s.fraction())
                if name is None:
                    name = 0 if len(series) == 1 else hash(component)

                self.experiments[experiment][name].append({'concentration': series,
                                                           'component': component,
                                                           'symbol': self.syms[component]})
            # drop names, no longer important
            submixes = self.experiments[experiment].values()
            self.expressions[experiment] = Expression(submixes)

    def layout(self, plate_size=(8, 12)):
        exp = self.expressions['experiment 2']
        layout = {}
        for experiment, exp in self.expressions.items():
            block_corners = []
            # prefer spacing of 1 unless 0 saves plates
            for spacing in range(2):
                block_size = np.array(exp.split_size[-2:]) + spacing
                num_blocks = int(np.prod(exp.split_size[:-2]))
                plate_tiling = np.floor(np.array(plate_size) / block_size)
                corners = [(np.floor(float(i) / plate_tiling[1]),
                            i % plate_tiling[1],
                            np.floor(float(i) / np.prod(plate_tiling))) for i in range(num_blocks)]
                block_corners.append(np.array(corners) * np.array(list(block_size) + [1]))

            layout[experiment] = block_corners[0] if block_corners[0][-1][2] < block_corners[1][-1][2] \
                else block_corners[1]
        return layout


class Expression(object):
    def __init__(self, submixes):
        """Represent experiment as symbolic expression. When fully expanded, each term represents a final reaction.
        Intermediate factorizations represent sub-mixes.
        :return:
        """
        self.lc_syms = []
        self.uc_syms = []
        self.split_size = []
        self.rank = []
        self.uc_fractions = {}
        self.lc_alphabet = (chr(i).lower() for i in range(65, 65 + 26))

        self.expression = None
        self.expression_eval = None
        self.components = {}
        self.h_values = {}

        self.define_symbols(submixes)
        self.sort()
        self.form_expression()
        self.pick_h2o()

    def define_symbols(self, submixes):
        for submix in submixes:
            # lowercase symbols indicate sub-mix in particular experiment
            self.lc_syms.append(sp.symarray(self.lc_alphabet.next(), len(submix[0]['concentration'])))
            # uppercase symbols indicate components, subscripts
            self.uc_syms.append([symbol_array(s['symbol'], len(s['concentration']), {'positive': True})
                                 for s in submix])
            # fraction of the final reaction associated with each sub-mix component
            fractions = {n[i]: c.fraction()
                         for s, n in zip(submix, self.uc_syms[-1])
                         for i, c in enumerate(s['concentration'])}
            self.uc_fractions.update(fractions)
            self.rank.append(max([s['component'].rank for s in submix]))

    def form_expression(self):
        # form the dictionary relating lowercase symbols to uppercase and water
        skip_h = False
        for l_terms, U_terms in zip(self.lc_syms, self.uc_syms):
            for i, (l, U) in enumerate(reversed(zip(l_terms, zip(*U_terms)))):
                if i == 0 and skip_h:
                    self.components[l] = sp.exp(sum([V for V in U]))
                else:
                    self.components[l] = sp.exp(sum([V for V in U] +
                                                    [sp.symbols('h_' + str(l), positive=True)]))
            skip_h = True

        self.expression = sp.prod([sum(s) for s in self.lc_syms])
        self.expression_eval = sp.expand(self.expression).subs(self.components)

    def pick_h2o(self):
        """Determine water added in each submix by solving constraints on added water.
        :return:
        """
        h_constraints = []
        master_equation = self.expression_eval.subs(self.uc_fractions)
        for term in master_equation.args:
            h_constraints.append(sp.Eq(sp.log(term).expand().nsimplify(), 1))

        # set free water values to zero
        self.h_values = {}
        # form system of linear equations
        h_syms = sp.Matrix(list(master_equation.free_symbols))
        M = sp.Matrix([h.args[0].coeff(1) for h in h_constraints]).jacobian(h_syms)
        x = [float(1 - (h.args[0] - h.args[0].coeff(1))) for h in h_constraints]
        self.h_values = {h: value for h, value in zip(h_syms, lstsq(M, x)[0])}
        # solved_constraints = sp.solve(h_constraints)
        # if type(solved_constraints) is list:
        # solved_constraints = solved_constraints[0]
        # for lhs, expr in solved_constraints.items():
        #     self.h_values.update({h: 0 for h in expr.free_symbols})
        #     self.h_values[lhs] = float(expr.subs(self.h_values))

    def get_submix(self, submix):
        """Retrieve submix corresponding to tuple. Depends on ordering of lc_syms.
        :param submix:
        :return:
        """
        split = sp.prod(self.lc_syms[i][j] for i, j in enumerate(submix))
        return sp.log(split.subs(self.components)).expand()

    def sort(self):
        """Reorder lc_syms based on highest rank of components and sub-mix size.
        :return:
        """
        # permute, sorting by lowest rank of split then size
        rank = [len(lc) if r == 0 else 1000 + r for r, lc in zip(self.rank, self.lc_syms)]
        self.lc_syms = [s for r, s in sorted(zip(rank, self.lc_syms), key=lambda x: x[0])]
        self.uc_syms = [s for r, s in sorted(zip(rank, self.uc_syms), key=lambda x: x[0])]
        self.split_size = tuple([len(l) for l in self.lc_syms])


class Instruction(object):
    def __init__(self):
        self.text = 'Add '
        self.split = (None, None)
        self.split_size = 0.
        self.split_volume = 0.
        self.split_label = 'M1'

    def get_text(self):
        text1 = 'Split each mix into %d sub-mixes with %.2g uL each.' % \
                (self.split_size, self.split_volume)

        mix_labels = ['%s-%d' % (self.split_size, i) for i in range(self.split_size)]
        text2 = 'Label the new sub-mixes %s.' % ', '.join(mix_labels)
        text3 = 'Add the following to each sub-mix:\n'
        pd.DataFrame()

    def get_html(self):
        pass


class Concentration(object):
    def __init__(self, input_string, stock=None):
        self.units = None
        self.value = 0.
        self.parse(input_string)
        self.stock = stock

    def parse(self, input_string):
        self.value, self.units = re.match(concentration_pattern, input_string).groups()[:2]
        self.value = float(self.value)
        self.units = self.units.strip()

    def fraction(self):
        if self.stock is None:
            return 1
        return self.value / self.stock.concentration.value

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
    def __init__(self, name, rank, concentration=None):
        self.name = name
        self.rank = rank
        self.concentration = Concentration('1X') if concentration is None else concentration

    def __repr__(self):
        return "%s" % self.name


def test():
    form = pd.read_table(test_form, sep=',')
    exp = Experiment(form, reaction_volume=test_reaction_volume,
                     pipette_loss=test_pipette_loss)
    csv.writer(open(output_csv, 'wb')).writerows(exp.layout())
    return exp


if __name__ == '__main__':
    test()