from flask import Flask, render_template, request, flash,url_for
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
import pandas as pd
import os, time
import numpy as np
from pandas import DataFrame as df


import predict

#a = pd.read_csv('./testdata.csv')
#preddata = a.iloc[:,5:49]

app = Flask(__name__)
app.secret_key = "shawroot"

class Newform(FlaskForm):
    submit1 = SubmitField('检测')
    submit2 = SubmitField('检测')
    submit3 = SubmitField('检测')
    submit4 = SubmitField('检测')
    submit5 = SubmitField('检测')
    submit6 = SubmitField('检测')
    submit7 = SubmitField('检测')
    submit8 = SubmitField('检测')
    submit9 = SubmitField('检测')


#app.config['JSON_AS_ASCII'] = False
@app.route("/",methods=['GET','POST'])
def idnex():
    form = Newform()
    if request.method == 'GET':
        a = pd.read_csv('testdata.csv')
        data = {
            '序号':a['序号'],             #0
            '任务名称':a['任务名称'],      #1
            '执行时间':a['执行时间'],      #2
            '执行结果':a['执行结果'],      #3
            '类型':a['类型']             #4
        }
        return render_template('service.html', data=data,form = form)

    if request.method == 'POST':

        if form.submit1.data:
            b = pd.read_csv('testdata.csv')
            b.iloc[0, 2] = time.strftime("%Y-%m-%d %H:%M:%S")
            preddata = np.array(b.iloc[0,5:126]).reshape(1,-1)
            result = predict.pred(preddata)
            b.iloc[0,3] = '已执行'
            b.iloc[0,4] = result
            b.to_csv('testdata.csv',index=False,encoding="utf_8_sig")
            data = {
                '序号': b['序号'],
                '任务名称': b['任务名称'],
                '执行时间': b['执行时间'],
                '执行结果': b['执行结果'],
                '类型': b['类型']
            }
            return render_template('service.html', data=data,form = form)

        elif form.submit2.data:
            b = pd.read_csv('testdata.csv')
            b.iloc[1, 2] = time.strftime("%Y-%m-%d %H:%M:%S")
            preddata = np.array(b.iloc[1,5:126]).reshape(1,-1)
            result = predict.pred(preddata)
            b.iloc[1,3] = '已执行'
            b.iloc[1,4] = result
            b.to_csv('testdata.csv', index=False,encoding="utf_8_sig")
            data = {
                '序号': b['序号'],
                '任务名称': b['任务名称'],
                '执行时间': b['执行时间'],
                '执行结果': b['执行结果'],
                '类型': b['类型']
            }
            return render_template('service.html', data=data,form = form)

        elif form.submit3.data:
            b = pd.read_csv('testdata.csv')
            b.iloc[2, 2] = time.strftime("%Y-%m-%d %H:%M:%S")
            preddata = np.array(b.iloc[2, 5:126]).reshape(1,-1)
            result = predict.pred(preddata)
            b.iloc[2,3] = '已执行'
            b.iloc[2,4] = result
            b.to_csv('testdata.csv', index=False,encoding="utf_8_sig")
            data = {
                '序号': b['序号'],
                '任务名称': b['任务名称'],
                '执行时间': b['执行时间'],
                '执行结果': b['执行结果'],
                '类型': b['类型']
            }
            return render_template('service.html', data=data,form = form)

        elif form.submit4.data:
            b = pd.read_csv('testdata.csv')
            b.iloc[3, 2] = time.strftime("%Y-%m-%d %H:%M:%S")
            preddata = np.array(b.iloc[3, 5:126]).reshape(1,-1)
            result = predict.pred(preddata)
            b.iloc[3,3] = '已执行'
            b.iloc[3,4] = result
            b.to_csv('testdata.csv', index=False,encoding="utf_8_sig")
            data = {
                '序号': b['序号'],
                '任务名称': b['任务名称'],
                '执行时间': b['执行时间'],
                '执行结果': b['执行结果'],
                '类型': b['类型']
            }
            return render_template('service.html', data=data,form = form)

        elif form.submit5.data:
            b = pd.read_csv('testdata.csv')
            b.iloc[4, 2] = time.strftime("%Y-%m-%d %H:%M:%S")
            preddata = np.array(b.iloc[4, 5:126]).reshape(1,-1)
            result = predict.pred(preddata)
            b.iloc[4,3] = '已执行'
            b.iloc[4,4] = result
            b.to_csv('testdata.csv', index=False,encoding="utf_8_sig")
            data = {
                '序号': b['序号'],
                '任务名称': b['任务名称'],
                '执行时间': b['执行时间'],
                '执行结果': b['执行结果'],
                '类型': b['类型']
            }
            return render_template('service.html', data=data,form = form)
        '''
        elif form.submit6.data:
            b = pd.read_csv('testdata.csv')
            b.iloc[5, 2] = time.strftime("%Y-%m-%d %H:%M:%S")
            preddata = np.array(b.iloc[5, 5:126]).reshape(1,-1)
            result = predict.pred(preddata)
            b.iloc[5,3] = '已执行'
            b.iloc[5,4] = result
            b.to_csv('testdata.csv', index=False,encoding="utf_8_sig")
            data = {
                '序号': b['序号'],
                '任务名称': b['任务名称'],
                '执行时间': b['执行时间'],
                '执行结果': b['执行结果'],
                '类型': b['类型']
            }
            return render_template('service.html', data=data,form = form)

        elif form.submit7.data:
            b = pd.read_csv('testdata.csv')
            b.iloc[6, 2] = time.strftime("%Y-%m-%d %H:%M:%S")
            preddata = np.array(b.iloc[6, 5:126]).reshape(1,-1)
            result = predict.pred(preddata)
            b.iloc[6,3] = '已执行'
            b.iloc[6,4] = result
            b.to_csv('testdata.csv', index=False,encoding="utf_8_sig")
            data = {
                '序号': b['序号'],
                '任务名称': b['任务名称'],
                '执行时间': b['执行时间'],
                '执行结果': b['执行结果'],
                '类型': b['类型']
            }
            return render_template('service.html', data=data,form = form)

        elif form.submit8.data:
            b = pd.read_csv('testdata.csv')
            b.iloc[7, 2] = time.strftime("%Y-%m-%d %H:%M:%S")
            preddata = np.array(b.iloc[7, 5:126]).reshape(1,-1)
            result = predict.pred(preddata)
            b.iloc[7,3] = '已执行'
            b.iloc[7,4] = result
            b.to_csv('testdata.csv', index=False,encoding="utf_8_sig")
            data = {
                '序号': b['序号'],
                '任务名称': b['任务名称'],
                '执行时间': b['执行时间'],
                '执行结果': b['执行结果'],
                '类型': b['类型']
            }
            return render_template('service.html', data=data,form = form)

        elif form.submit9.data:
            b = pd.read_csv('testdata.csv')
            b.iloc[8, 2] = time.strftime("%Y-%m-%d %H:%M:%S")
            preddata = np.array(b.iloc[8, 5:126]).reshape(1,-1)
            result = predict.pred(preddata)
            b.iloc[8,3] = '已执行'
            b.iloc[8,4] = result
            b.to_csv('testdata.csv', index=False,encoding="utf_8_sig")
            data = {
                '序号': b['序号'],
                '任务名称': b['任务名称'],
                '执行时间': b['执行时间'],
                '执行结果': b['执行结果'],
                '类型': b['类型']
            }
            return render_template('service.html', data=data,form = form)
        '''

        return render_template('service.html')


if __name__ == '__main__':
    app.run()