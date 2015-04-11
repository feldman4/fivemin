from flask import Flask, render_template, url_for, session, request, redirect, g, flash
import fivemin
import pandas as pd
import sys
from bs4 import BeautifulSoup

app = Flask(__name__)
app.config['DEBUG'] = True
app.config['PROPAGATE_EXCEPTIONS'] = True
app.secret_key = 'mr. secrets'

@app.route('/')
def index():
    return render_template('input.html')

@app.route('/<planet>')
def eatshit(planet):
    return 'Welcome to %s earthling' % planet


@app.route('/changed')
def changed():
    print render_template('input.html', layout=session['layout'])
    return render_template('input.html', layout=session['layout'])


@app.route('/signup', methods=['POST'])
def signup():
    form = request.json['data']
    try:
        df = pd.DataFrame(form[1:], columns=form[0])
        df = df.replace({None: float('nan')}).dropna()
        df = df.replace({'': float('nan')})
        print df
        experiment = fivemin.Experiment(df, reaction_volume=10, pipette_loss=1.1)
        if experiment.error_flag is False:

            experiment.write_instructions()
            experiment.layout2()
            plates = experiment.layout.plates_html()

            instr = ''.join(['<li class="instruction">' + a + '</li>' for a in experiment.print_instructions(mode='html')])
            instr = '<ol class="instruction">' + instr + '</ol>'
        else:
            instr = '<p>Error in following rows:\n' + \
                    experiment.error_rows.fillna('-').to_html() + '</p>'
            plates = []
    # catch-all to prevent server from crashing
    except:
        plates = []
        instr = '<p> invalid form </p>'
        print "Unexpected error:", sys.exc_info()[0]
        raise
    return render_template('layout.html', plates=plates, instructions=instr)


@app.errorhandler(404)
def page_not_found(e):
    """Return a custom 404 error."""
    return 'Sorry, nothing at this URL.', 404


if __name__ == '__main__':
    app.run()
