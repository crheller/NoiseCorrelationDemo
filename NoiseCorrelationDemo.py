from bokeh.layouts import column, grid, row
from bokeh.models import Button
from bokeh.palettes import RdYlBu3
from bokeh.plotting import figure, curdoc
from bokeh.models import Button, ColumnDataSource, ColorBar
from bokeh.models.glyphs import Line
from bokeh.models.widgets import Slider, TextInput, DataTable, TableColumn
import helpers as help
import numpy as np

# generate initial data set
u1 = [2, 2]
u2 = [4, 4]
cov = np.array([[1, 0], [0, 1]])
x = help.generate_data(u1, cov)
y = help.generate_data(u2, cov)

# get the decoding axes and dprime values for this starting point
values = np.round(help.get_table_values(x, y), 3)
metrics = ['dprime_pop_LDA', 'dprime_pop_NULL', 'dprime_ind', 'ratio_LDA', 'ratio_NULL', 'cos(NULL, Noise)',
            'cos(LDA, Noise)', 'rsc']

decoding_metrics = dict(metrics=metrics, values=values)
source = ColumnDataSource(decoding_metrics)
columns = [
        TableColumn(field="metrics", title="Metric"),
        TableColumn(field="values", title="Value"),
    ]
data_table = DataTable(source=source, columns=columns, width=400, height=280)

# set the date source for two stimuli/signals that are trying to be discriminated
source_s1 = ColumnDataSource(dict(x=x[:, 0], y=x[:, 1]))
source_s2 = ColumnDataSource(dict(x=y[:, 0], y=y[:, 1]))

# set up plot layout
p = figure(x_range=(0, 10), y_range=(0, 10), toolbar_location=None)
p.outline_line_color = 'gray'
p.grid.grid_line_color = None
p.title.text = "NoiseCorrelationDemo"
p.xaxis.axis_label = "Neuron 1"
p.yaxis.axis_label = "Neuron 2"

# plot the initial data
s1 = p.circle(x='x', y='y', size=4, color="red", alpha=0.5, source=source_s1)
s2 = p.circle(x='x', y='y', size=4, color="blue", alpha=0.5, source=source_s2)

# plot initial decoding axes
NULL = help.get_null_axis(x, y)
LDA = help.get_LDA_axis(x, y)
noise = help.get_noise_PC(x, y)

NULL_x = [x.mean(axis=0)[0], y.mean(axis=0)[0]]
NULL_y = [x.mean(axis=0)[1], y.mean(axis=0)[1]]
NULL_source = ColumnDataSource(dict(x=NULL_x, y=NULL_y))

uall = np.concatenate((x, y), axis=0).mean(axis=0)
LDA_x = [uall[0]-LDA[0], uall[0]+LDA[0]]
LDA_y = [uall[1]-LDA[1], uall[1]+LDA[1]]
LDA_source = ColumnDataSource(dict(x=LDA_x, y=LDA_y))

noise_x = [uall[0]-noise[0], uall[0]+noise[0]]
noise_y = [uall[1]-noise[1], uall[1]+noise[1]]
noise_source = ColumnDataSource(dict(x=noise_x, y=noise_y))

p.line(x="x", y="y", line_color="black", line_width=3, legend='NULL axis', source=NULL_source)
p.line(x="x", y="y", line_color="gold", line_width=3, legend='LDA axis', source=LDA_source)
p.line(x="x", y="y", line_color="purple", line_width=3, legend='noise axis', source=noise_source)
# set up widgets for editing means / covariance
# Set up widgets
covariance = Slider(title="covariance", value=0.0, start=-1.0, end=1.0, step=0.01)
n1_variance = Slider(title="neuron 1 variance", value=1.0, start=0.0, end=3.0, step=0.01)
n2_variance = Slider(title="neuron 2 variance", value=1.0, start=0.0, end=3.0, step=0.01)
n1_u1 = Slider(title="Mean neuron 1", bar_color='red', value=3.0, start=2.0, end=8.0, step=0.01)
n1_u2 = Slider(title="Mean neuron 1", bar_color='blue', value=5.0, start=2.0, end=8.0, step=0.01)
n2_u1 = Slider(title="Mean neuron 2", bar_color='red', value=3.0, start=2.0, end=8.0, step=0.01)
n2_u2 = Slider(title="Mean neuron 2", bar_color='blue', value=5.0, start=2.0, end=8.0, step=0.01)

def update_table(x, y):
    # update data table
    source.data = {"metrics": metrics, "values": np.round(help.get_table_values(x, y), 3)}

def update_axes(x, y):
    x_norm, y_norm = help.normalize_variance(x, y)
    NULL = help.get_null_axis(x_norm, y_norm)
    LDA = help.get_LDA_axis(x_norm, y_norm)
    print(np.dot(LDA, NULL))
    noise = help.get_noise_PC(x_norm, y_norm)

    NULL_x = [x.mean(axis=0)[0], y.mean(axis=0)[0]]
    NULL_y = [x.mean(axis=0)[1], y.mean(axis=0)[1]]

    LDA *= (np.linalg.norm([[x.mean(axis=0)[0], y.mean(axis=0)[0]],
                            [x.mean(axis=0)[1], y.mean(axis=0)[1]]])/2)
    NULL *= (np.linalg.norm([[x.mean(axis=0)[0], y.mean(axis=0)[0]],
                            [x.mean(axis=0)[1], y.mean(axis=0)[1]]])/2)
    noise *= (np.linalg.norm([[x.mean(axis=0)[0], y.mean(axis=0)[0]],
                            [x.mean(axis=0)[1], y.mean(axis=0)[1]]])/2)
    uall = np.concatenate((x, y), axis=0).mean(axis=0)
    LDA_x = [uall[0]-LDA[0], uall[0]+LDA[0]]
    LDA_y = [uall[1]-LDA[1], uall[1]+LDA[1]]
    noise_x = [uall[0]-noise[0], uall[0]+noise[0]]
    noise_y = [uall[1]-noise[1], uall[1]+noise[1]]
    NULL_x = [uall[0]-NULL[0], uall[0]+NULL[0]]
    NULL_y = [uall[1]-NULL[1], uall[1]+NULL[1]]

    NULL_source.data = dict(x=NULL_x, y=NULL_y)
    LDA_source.data = dict(x=LDA_x, y=LDA_y)
    noise_source.data = dict(x=noise_x, y=noise_y)

def update_data(attrname, old, new):
    x = covariance.value
    n1 = n1_variance.value
    n2 = n2_variance.value
    u11 = n1_u1.value
    u12 = n1_u2.value
    u21 = n2_u1.value
    u22 = n2_u2.value

    # update stim1
    u = [u11, u21]
    cov = np.array([[n1, x], [x, n2]])
    data1 = help.generate_data(u, cov)
    source_s1.data = dict(x=data1[:, 0], y=data1[:, 1])

    # update stim2
    u = [u12, u22]
    cov = cov
    data2 = help.generate_data(u, cov)
    source_s2.data = dict(x=data2[:, 0], y=data2[:, 1])

    update_axes(data1, data2)

    update_table(data1, data2)


for w in [covariance, n1_variance, n2_variance, n1_u1, n1_u2, n2_u1, n2_u2]:
    w.on_change('value', update_data)

inputs = column(covariance, n1_variance, n2_variance, n1_u1, n1_u2, n2_u1,
                n2_u2, data_table)

curdoc().add_root(row(inputs, p))
