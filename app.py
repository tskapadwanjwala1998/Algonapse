from flask import Flask,render_template,request,jsonify
from random import sample
import pandas as pd
import gatherer
import algo

app = Flask(__name__)



@app.route('/')
def index():
    return render_template('main.html')

@app.route('/data',methods=["POST", 'GET'])
def data():
    
    if request.method=='POST':
        print(request)
        # if symbol != request.form['search']:
        symbol = request.form['search']
        source = request.form['model_name']
        gatherer.data(symbol,source)
        algo.task2(symbol,source)
        output = algo.task3(symbol,source)
        return chart1(output)
    else:
         return render_template('main.html')
 
@app.route('/chart1')
def chart1(output):
    
    print(output)
    return render_template('security_analysis.html', analysis_list=output)


if __name__ == '__main__':
    app.run(debug=1)




