import re
import pandas as pd
from collections import defaultdict
import numpy as np
import sympy as sp
from numpy.linalg import lstsq
import sys
from time import gmtime, strftime

test_form = 'reaction_tests.csv'
output_csv = 'output2.csv'
concentration_pattern = '([0-9]*\.*[0-9]*)(.*)'
named_series_pattern = '((.*):)*(.*)'
test_reaction_volume = 10
test_pipette_loss = 1.1
pd.options.display.max_columns = 30


def symbol_array(sym, size, assumptions=None):
    return [sp.symbols(sym + '_' + str(i), **assumptions)
            for i in range(size)]


# noinspection PyTypeChecker
class Experiment(object):
    def __init__(self, form, reaction_volume=10, pipette_loss=1., plate_size=(8, 12)):
        """Container for experiment, including all parameters and generated information.
        :param form:
        :return:
        """
        self.instructions = []
        self.components = []
        self.series = {}
        self.reactions = {}
        self.reaction_volume = reaction_volume
        self.pipette_loss = pipette_loss
        self.water = Component('water', 0)
        self.syms = {self.water: 'h'}
        self.syms_to_components = {}
        self.expression = None
        self.uc_alphabet = (chr(i) for i in range(65, 65 + 26))

        self.layout = None
        self.plate_size = plate_size

        self.setup(form)
        strftime("%Y-%m-%d %H:%M:%S", gmtime())

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
        self.series = defaultdict(list)
        # determine sub-mixes and components
        # TODO convert concentrations to fold dilutions
        for entry, component in zip(form.fillna(value='1X')['experiment'], self.components):
            # generic format A: 1, 2, 3
            name, series = re.match(named_series_pattern, entry).groups()[1:3]
            series = [Concentration(c.strip(), stock=component) for c in series.split(',')]
            series = sorted(series, key=lambda s: s.fraction())
            if name is None:
                name = 0 if len(series) == 1 and component.rank == 0 else hash(component)

            self.series[name].append({'concentration': series,
                                      'component': component,
                                      'symbol': self.syms[component]})
        self.series = self.series.values()
        self.expression = Expression(self.series)
        self.series = [self.series[r] for r in self.expression.sorted_order]

    def layout2(self, mode='organized'):
        self.layout = Layout(self.expression, self.series, mode=mode)

    def layout(self, plate_size=(8, 12)):

        # find corners

        block_corners = []
        # prefer spacing of 1 unless 0 saves plates
        for spacing in range(2):
            block_size = np.array(self.expression.split_size[-2:]) + spacing
            num_blocks = int(np.prod(self.expression.split_size[:-2]))
            plate_tiling = np.floor(np.array(plate_size) / block_size)
            corners = [(np.floor(float(i) / plate_tiling[1]),
                        i % plate_tiling[1],
                        np.floor(float(i) / np.prod(plate_tiling))) for i in range(num_blocks)]
            block_corners.append(np.array(corners) * np.array(list(block_size) + [1]))

        layout = block_corners[0] if block_corners[0][-1][2] < block_corners[1][-1][2] \
            else block_corners[1]

        # generate compact, detailed DataFrames
        iterables = [['/'.join([str(k['concentration'][i]) for k in s])
                      for i in range(len(s[0]['concentration']))] for s in self.series]
        names = ['/'.join([str(k['component']) for k in s])
                 for s in self.series]
        iterables = [it for s, it in zip(self.series, iterables) if len(s[0]['concentration']) > 1]
        names = [n for s, n in zip(self.series, names) if len(s[0]['concentration']) > 1]
        compact = pd.MultiIndex.from_product(iterables, names=names)
        compact_df = pd.DataFrame(np.random.rand(np.prod(self.expression.split_size), 4), index=compact)
        return layout, compact_df

    def write_instructions(self):
        """Find volumes of each component in each submix, as well as volume of split between submixes. Use to create
        Instructions.
        :return:
        """

        pd.set_option('precision', 3)
        for i, count in enumerate(self.expression.split_size):
            this, vol = [], []
            for submix in [[0 for j in range(i)] + [k] for k in range(count)]:
                vol = np.prod(self.expression.split_size[i + 1:]) * self.reaction_volume * \
                      self.pipette_loss ** (len(self.expression.split_size) - (i + 1))
                if i == 0:
                    tmp = self.expression.expression_to_dict(self.expression.get_submix(submix))
                else:
                    tmp = self.expression.expression_to_dict(self.expression.get_submix(submix) -
                                                             self.expression.get_submix(submix[:-1]))
                # swap in component names
                this.append({self.syms_to_components[key]: val
                             for key, val in tmp.items()})
            # format into table, make water last if present
            tbl = pd.DataFrame(this) * vol
            reindex = list(tbl.columns)
            if self.water in reindex:
                reindex.append(reindex.pop(reindex.index(self.water)))
            tbl = tbl[reindex].transpose()
            tbl.columns = ['%d-%d' % (i + 1, j + 1) for j in tbl.columns]
            tbl[tbl == 0] = float('nan')
            split_vol = self.expression.get_split(i).subs(self.expression.loss, self.pipette_loss) * \
                        self.reaction_volume
            self.instructions.append(Instruction(tbl, split_vol))

    def print_instructions(self):
        lines = []
        for step, instr in enumerate(self.instructions):
            text = instr.get_first_text() if step == 0 else instr.get_text()
            lines.append('%d. ' % (step + 1) + text + '\n')
        return lines


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
        self.volumes = []
        self.uc_fractions = {}
        self.lc_alphabet = (chr(i).lower() for i in range(65, 65 + 25))
        self.loss = sp.symbols('z')
        self.sorted_order = []

        self.expression = None
        self.expression_eval = None
        self.components = {}
        self.h_values = {}
        print 'start of expr', strftime("%Y-%m-%d %H:%M:%S", gmtime())
        self.define_symbols(submixes)
        print 'start of sort', strftime("%Y-%m-%d %H:%M:%S", gmtime())
        self.sort()
        print 'start of form_expr', strftime("%Y-%m-%d %H:%M:%S", gmtime())
        self.form_expression()
        print 'start of pick_h2o', strftime("%Y-%m-%d %H:%M:%S", gmtime())
        self.pick_h2o()
        print 'end of pick_h2o', strftime("%Y-%m-%d %H:%M:%S", gmtime())

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
        print 'before eval', strftime("%Y-%m-%d %H:%M:%S", gmtime())
        self.expression_eval = sp.expand_mul(self.expression.xreplace(self.components))

    def pick_h2o(self):
        """Determine water added in each submix by solving constraints on added water.
        :return:
        """
        h_constraints = []
        print 'before the mas subs', strftime("%Y-%m-%d %H:%M:%S", gmtime())
        master_equation = self.expression_eval.xreplace(self.uc_fractions)
        print 'after the mas subs', strftime("%Y-%m-%d %H:%M:%S", gmtime())
        for term in master_equation.args:
            h_constraints.append(sp.Eq(sp.log(term).expand(), 1))
        print 'after the expandsimplify', strftime("%Y-%m-%d %H:%M:%S", gmtime())

        # set free water values to zero
        self.h_values = {}
        # form system of linear equations
        h_syms = sp.Matrix(list(master_equation.free_symbols))
        M = sp.Matrix([h.args[0].coeff(1) for h in h_constraints]).jacobian(h_syms)
        x = [float(1 - (h.args[0] - h.args[0].coeff(1))) for h in h_constraints]
        print 'ready for lstsq', strftime("%Y-%m-%d %H:%M:%S", gmtime())
        self.h_values = {h: value for h, value in zip(h_syms, lstsq(M, x)[0])}

        # determine final volume for each split
        self.volumes = []
        for i in range(len(self.lc_syms)):
            term = sp.prod(s[0] for s in self.lc_syms[:i + 1]).subs(self.components)
            term = sp.log(term).expand()
            self.volumes.append(term.subs(self.uc_fractions).subs(self.h_values))

    def get_submix(self, submix):
        """Retrieve submix corresponding to tuple. Depends on ordering of lc_syms.
        :param submix:
        :return:
        """
        split = sp.prod(self.lc_syms[i][j] for i, j in enumerate(submix))
        # TODO collect from main expression and count terms to get volume
        return sp.log(split.subs(self.components)).expand()

    def get_split(self, split):
        """Return expression for material in split from step i to step i+1.
        :param split:
        :return:
        """
        if split == 0:
            return self.loss ** (len(self.split_size) - 1) * np.prod(self.split_size)

        term = sp.prod(s[0] for s in self.lc_syms[:split]).subs(self.components).subs(self.h_values)
        term = sp.log(term).expand().subs(self.uc_fractions)
        return self.loss ** (len(self.split_size) - split - 1) * \
               np.prod(self.split_size[split + 1:]) * term

    def expression_to_dict(self, expression):
        result = {}
        complete_dict = self.uc_fractions.copy()
        complete_dict.update(self.h_values)
        it = expression.args
        # take care of expression with only one variable
        if len(it) == 0:
            it = [expression]
        for arg in it:
            letter = str(arg)[0]
            result.update({letter: complete_dict[arg]})
        return result

    def sort(self):
        """Reorder lc_syms based on highest rank of components and submix size.
        :return:
        """
        # permute, sorting by lowest rank of split then size
        rank = [len(lc) if r == 0 else 1000 + r for r, lc in zip(self.rank, self.lc_syms)]
        self.lc_syms = [s for r, s in sorted(zip(rank, self.lc_syms), key=lambda x: x[0])]
        self.uc_syms = [s for r, s in sorted(zip(rank, self.uc_syms), key=lambda x: x[0])]
        self.split_size = tuple([len(l) for l in self.lc_syms])
        self.sorted_order = [i for r, i in sorted(zip(rank, range(len(rank))))]


class Instruction(object):
    def __init__(self, table, split_volume):
        self.table = np.round(table, 2).replace(to_replace=0, value=float('nan'))
        self.split_volume = split_volume
        self.split_size = table.shape[1]
        self.text = 'Add '
        self.split = (None, None)
        self.split_size = 0.
        self.split_label = 'M1'
        self.plural = lambda i: '' if i == 1 else 'es'

    def get_text(self):
        count = self.table.shape[1]
        text1 = 'Transfer %.3g uL to each of %d submix%s.\n' % \
                (self.split_volume, count, self.plural(count))
        if count is 1:
            text1 = ''
        text2 = 'Add the following to each submix:\n'
        return text1 + text2 + '\n' + str(self.table.fillna('-'))

    def get_first_text(self):
        count = self.table.shape[1]
        text1 = 'Make %d mastermix%s with the following:\n' % (count, self.plural(count))
        return text1 + '\n' + str(self.table.fillna('-'))

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
        return "%.2g%s" % (self.value, self.units)


class Component(object):
    def __init__(self, name, rank, concentration=None):
        self.name = name
        self.rank = rank
        self.concentration = Concentration('1X') if concentration is None else concentration

    def __repr__(self):
        return "%s" % self.name


class Layout(object):
    def __init__(self, expression, series, mode='organized', plate_size=(8, 12)):
        """Organize reactions into a plate format, based around blocks of size (m x n), where m, n
        are the last two non-singleton splits. In organized mode, group blocks into rows and columns by preceding
        splits. In compact mode, pack together blocks efficiently without regard for rows and columns.
        :param Expression expression:
        :param components:
        :param str mode:
        :return:
        """
        self.mode = mode
        self.expression = expression
        self.series = series
        self.table = None
        self.block_size = []
        self.split_size = expression.split_size
        self.plate_size = plate_size
        if mode is 'organized':
            self.layout_organized()
        elif mode is 'compact':
            self.layout_compact()

    def layout_organized(self):
        lens = lambda x: len(x[0]['concentration'])
        # define basic block from last two non-singleton splits
        non_singletons = [s for s in reversed(self.series) if lens(s) > 1]
        self.block_size = ([lens(s) for s in non_singletons] + [1, 1])[:2]
        # get list of preceding non-singleton dimensions
        preceding = non_singletons[2:]
        # order in rows/columns to minimize plate usage
        best = ([], np.prod(self.block_size))
        for i in range(2 ** len(preceding)):
            order = [int(s) for s in "{0:b}".format(i)]
            block_size = list(self.block_size)
            best_order = []
            for j, (split, row_col) in enumerate(zip(preceding, order)):
                best_order.append(row_col)
                block_size[row_col] *= lens(split)
                overflow = any([b > p for b, p in zip(block_size, self.plate_size)])
                if overflow or j == len(preceding) - 1:
                    if overflow:
                        block_size[row_col] /= lens(split)
                        best_order.pop()
                    if np.prod(block_size) > best[1]:
                        best = (best_order, np.prod(block_size))
                        continue

        names = lambda x: ['|'.join([str(z['component']) for z in y]) for y in x]
        conc = lambda x: [['|'.join([str(z['concentration'][i]) for z in y])
                           for i in range(len(y[0]['concentration']))] for y in x]
        row_it = [non_singletons[0]] + [p for p, b in zip(preceding, best[0]) if not b]
        col_it = [non_singletons[1]] + [p for p, b in zip(preceding, best[0]) if b]

        row_index = pd.MultiIndex.from_product(conc(row_it), names=names(row_it))
        col_index = pd.MultiIndex.from_product(conc(col_it), names=names(col_it))

        base_df = pd.DataFrame(0, index=row_index, columns=col_index)
        # figure out number of plates and DataFrame on each plate
        num_plates = np.prod(self.split_size) / best[1]
        plate_splits = preceding[len(best[0]):]
        # construct on-plate DataFrame from two MultiIndex
        plate_dfs = []
        plate_names = names(plate_splits)
        for plate in range(num_plates):
            subsplit = [lens(p) for p in plate_splits]
            plate_vals = [plate % np.prod(subsplit[:i+1]) for i in range(len(plate_splits))]
            this_conc = ['|'.join([str(z['concentration'][v]) for z in y])
                         for y, v in zip(plate_splits, plate_vals)]
            this_col_index = pd.MultiIndex.from_product(this_conc + conc(col_it),
                                                        names=plate_names + names(col_it))
            plate_dfs.append(pd.DataFrame(0, index=row_index, columns=this_col_index))

        self.plate_dfs = plate_dfs

        # # format using MultiIndex
        # # generate compact, detailed DataFrames
        # iterables = [['/'.join([str(k['concentration'][i]) for k in s])
        #               for i in range(len(s[0]['concentration']))] for s in self.series]
        # names = ['/'.join([str(k['component']) for k in s])
        #          for s in self.series]
        # iterables = [it for s, it in zip(self.series, iterables) if len(s[0]['concentration']) > 1]
        # names = [n for s, n in zip(self.series, names) if len(s[0]['concentration']) > 1]
        # compact = pd.MultiIndex.from_product(iterables, names=names)


    def layout_compact(self):
        pass


def test(filename=test_form):
    form = pd.read_table(filename, sep=',')
    print strftime("%Y-%m-%d %H:%M:%S", gmtime())
    exp = Experiment(form, reaction_volume=test_reaction_volume,
                     pipette_loss=test_pipette_loss)

    exp.write_instructions()
    exp.layout2()
    layout = exp.layout.plate_dfs
    with open(filename[:-4] + '_output.txt', 'w') as fh:
        fh.writelines([a + '\n' for a in exp.print_instructions()])
        fh.writelines([df.to_string() + '\n\n' for df in layout])

    return exp


if __name__ == '__main__':
    test(filename=sys.argv[1])
