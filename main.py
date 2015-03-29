from flask import Flask, render_template
import fivemin

app = Flask(__name__)
app.config['DEBUG'] = True

# Note: We don't need to call run() since our application is embedded within
# the App Engine WSGI application server.


@app.route('/')
def hello2():
    """Return a friendly HTTP greeting."""
    return 'Fuck You Worlds!'

@app.route('/<planet>')
def eatshit(planet):
    return 'Welcome to %s earthling' % planet

@app.route('/hello/<name>')
def hello(name=None):
    return render_template('hello.html', name=name)


@app.errorhandler(404)
def page_not_found(e):
    """Return a custom 404 error."""
    return 'Sorry, nothing at this URL.', 404

if __name__ == '__main__':
    app.run()