
from __future__ import print_function

import torch
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.transforms as transforms
from celluloid import Camera
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm
import matplotlib.colorbar
import matplotlib.colors

from pathlib import Path
dir_path=Path(__file__).resolve().parent
FOLDER_OUTPUT=str(dir_path.parent.parent)+"/visualizations/"


global CAMPERO_RULES_ARITIES
global CAMPERO_RULES_TEMPLATES
CAMPERO_RULES_ARITIES=[[1,1,0], [1,1,2], [2,2,2], [1,2,0], [2,2,0], [0,0,0], [0,0,0], [2,1, 0], [2,2,0], [2,2,2], [2,2,1], [1,2,0], [1,2,1], [1,2,2], [2,2,2],[2,2,2]]#CHANGE THIS BACK FOR TWO EDGE...
CAMPERO_RULES_TEMPLATES=["F(X) <-- F(X)", "F(x)<--F(Z),F(Z,X)", "F(x,y)<-- F(x,Z),F(Z,Y)", "F(X) <-- F(X,X)", "F(X,Y) <-- F(X,Y)", "", "", "F(X,X) <-- F(X)", "F(x,y) <-- F(y,x)", "F(x,y)<---F(y,Z),F(X,Z)", "F(x,y)<-- F(y,x),F(x)", "F(X) <-- F(X,Z)", "F(X) <-- F(X,Z), F(Z)","F(X) <-- F(X,Z), F(X,Z)", "F(X,X) <-- F(X,Z), F(X,Z)","F(X,Y) <-- F(X,Y), F(X,Y)"]
assert len(CAMPERO_RULES_ARITIES)==len(CAMPERO_RULES_TEMPLATES)


##-----For visualization (+++)----------------------   
def visualize_parameters(valuation, rules_str, background_predicates, intensional_predicates, predicates_labels, model):
    """
    Parameters for visualization, such as masks and labels
    """
    #define rule labels for visualization
    rules_labels=["R{}, {}".format(i, CAMPERO_RULES_TEMPLATES[rules_str[i]-1]) for i in range(len(rules_str))]
    #rules_str indicate template number, has to remove 1
    rules_arity=[CAMPERO_RULES_ARITIES[r-1] for r in rules_str]
    predicate_arity=[]
    num_predicates=len(background_predicates)+len(intensional_predicates)
    num_rules=len(rules_labels)
    for predicate in range(num_predicates):
        #unary predicate
        if valuation[predicate].size()[1] == 1:
            predicate_arity.append(1)
        #binary predicate
        else:
            predicate_arity.append(2)
    
    #define mask for visualisation
    if model=="default" or model=="model-sim":
        num_elt, offset=3,0
    elif model=="model-one" or model=="model-h":
        num_elt, offset=2,1

    mask=torch.ones((num_rules, num_elt, num_predicates))
    for predicate in range(num_predicates):#TODO: VECTORISE
        for body in range(num_elt):
            for rule in range(num_rules):
                # Mask of 0 if predicat not good arity compared to body rule or if body rule does not exist
                if rules_arity[rule][body+offset]==0 or not(rules_arity[rule][body+offset]==predicate_arity[predicate]):
                    mask[rule,body,predicate]=0
    mask_bool=(mask==0)#data will not be shown in visualisation where mask is true
    mask_bool_np=mask_bool.transpose(1,2).detach().numpy() #shape num_rules, num_pred, 3)
    #embedding labels to visualize embeddings
    embedding_labels=predicates_labels+np.repeat(rules_labels,3).tolist()
    return mask_bool_np, rules_labels, embedding_labels
##-----------------------------------------------



def visualize(data, losses=None, y_labels=None, title_plot="Plot", title_subplots=None, mask=None):
    """
    Animate the attention weights, how they change over time. 

    Inputs:
        data tensor size (num_rules, 3, num_pred)
        y_labels may be None else name of the predicates ideally!
        title plot is title of the plot...
    Output:
        ANIMATE...
    """
    #(0) PRELIMNINARIES
    num_rules, three, num_pred=data.shape
    assert three==3
    data_np=data.transpose(1,2).detach().numpy() #SIZE (num_rules, num_pred, 3)
    x_labels = ["h", "b1", "b2"]
    ax=[]

    #(1) Create figure Instance, and one subplot for each rule
    fig = plt.figure(figsize = (num_rules*2, num_pred/3)) # width x height
   
    gs = fig.add_gridspec(1, num_rules)

    for rule in range(num_rules):
        ax.append(fig.add_subplot(gs[0,rule])) #(num_pred, 3, rule+1)# row, column, position
    

    for rule in range(num_rules):
        plt.yticks(rotation = 40)
        mask_temp=None
        if mask is not None:
            mask_temp=mask[rule, :,:]
        sns.heatmap(data= data_np[rule, :, :], ax=ax[rule], cmap = "YlGnBu", cbar=False, vmin=0, vmax=1, square=True, mask=mask_temp)
  
    for rule in range(num_rules):
        if title_subplots is not None:
            ax[rule].set_title(title_subplots[rule],fontsize=7)
        if x_labels is not None:
            ax[rule].set_xticks(np.arange(len(x_labels)))
            ax[rule].set_xticklabels(x_labels, fontsize=5)
        if y_labels is not None:
            ax[rule].set_yticks(np.arange(len(y_labels)))
            ax[rule].set_yticklabels(y_labels, fontsize=5, rotation="horizontal")

    plt.tight_layout()
    plt.yticks(rotation=90)
    #plt.savefig("sample.jpg")
    plt.show()

def animate(data, losses=None, y_labels=None, x_labels = None, title_plot="Plot", title_subplots=None, scale=1, mask=None):
    """
    Animate the attention weights, how they change over time. 

    Inputs:
        data tensor size (TIME_STEPS, num_rules, 3, num_pred)
        losses: possibly add losses for each steps
        y_labels may be None else name of the predicates ideally!
        title plot is title of the plot...
    Output:
        Gif animation, saved in path_output.
    """
    if len(list(data.shape))==4:
        animate_unifs(data, losses, y_labels, x_labels, title_plot, title_subplots, scale, mask)
    elif len(list(data.shape))==3:
        animate_embeddings(data, losses, y_labels, x_labels, title_plot, scale)
    else:
        raise NotImplementedError
    


def animate_embeddings(data, losses=None, y_labels=None, x_labels = None, title_plot="Plot", scale=1):
    """
    Animate the attention weights, how they change over time. 

    Inputs:
        data tensor size (TIME_STEPS, num_rules, 3, num_pred)
        losses: possibly add losses for each steps
        y_labels may be None else name of the predicates ideally!
        title plot is title of the plot...
    Output:
        Gif animation, saved in path_output.
    """

    #(0) PRELIMNINARIES
    time_steps, num_embeddings, num_features=data.shape
    data_np=data.detach().numpy()#SIZE (TIME_STEPS, num_embeddings, num_features)
      
    #(1) Create figure Instance, and one subplot for each rule
    fig = plt.figure(figsize = (num_embeddings, num_features)) # width x height
    gs = fig.add_gridspec(1, 1)#useless
    ax=fig.add_subplot(gs[0,0])
    sns.heatmap(data= data_np[0,:, :], ax=ax, cmap = "YlGnBu", cbar=False, vmin=0, vmax=1, square=True)

    if y_labels is not None:
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels, fontsize=5)

    camera = Camera(fig)

    # (2) Reuse the figure and after each frame is created, take a snapshot with the camera.
    for t in range(time_steps):
        #NOTE: Animating the title did not work. As a workaround, create a text object:
        mark_time = " T=" + str(scale*t)
        if losses is not None and t<len(losses):
            mark_time+=" Loss="+str(round(losses[t],3))#add loss info if given
        ax.text(-0.5, -0.1, mark_time, transform=ax.transAxes)
        sns.heatmap(data_np[t, :, :], ax=ax, cmap="YlGnBu", cbar=False,vmin=0, vmax=1, square=True)  # cmap="YlGnBu"
        if y_labels is not None:
            ax.set_yticks(np.arange(len(y_labels)))
            ax.set_yticklabels(y_labels, fontsize=5, rotation="horizontal")
        camera.snap()

    # (3) After all frames have been captured, create the animation.
    animation = camera.animate()
    path_output=FOLDER_OUTPUT+title_plot+".gif"
    animation.save(path_output)

test_data=torch.rand((3, 3, 5))
test_visu=animate_embeddings(test_data, losses=None, y_labels=["p1", "p2", "r2"], title_plot="Plot", scale=1)


def animate_unifs(data, losses=None, y_labels=None, x_labels = None, title_plot="Plot", title_subplots=None, scale=1, mask=None):
    """
    Animate the attention weights, how they change over time. 

    Inputs:
        data tensor size (TIME_STEPS, num_rules, 3, num_pred)
        losses: possibly add losses for each steps
        y_labels may be None else name of the predicates ideally!
        title plot is title of the plot...
    Output:
        Gif animation, saved in path_output.
    """

    #(0) PRELIMNINARIES
    time_steps, num_rules, rule_size, num_pred=data.shape
    data_np=data.transpose(2,3).detach().numpy()#SIZE (TIME_STEPS, num_rules, num_pred, 3)
    ax=[]
    if y_labels is None:#ELSE MAY PROVIDE Y LABELS
        y_labels=["p_"+str(i) for i in range(num_pred)]
    
    #(1) Create figure Instance, and one subplot for each rule
    fig = plt.figure(figsize = (num_rules*2, num_pred/2)) # width x height
    gs = fig.add_gridspec(1, num_rules)
    for rule in range(num_rules):
        ax.append(fig.add_subplot(gs[0,rule]))#(num_pred, 3, rule+1)) 
        
    for rule in range(num_rules):
        mask_temp=None
        if mask is not None:
            mask_temp=mask[rule, :,:]
        sns.heatmap(data= data_np[0, rule, :, :], ax=ax[rule], cmap = "YlGnBu", cbar=False, vmin=0, vmax=1, square=True, mask=mask_temp)

    for rule in range(num_rules):
        if title_subplots is not None:
            ax[rule].set_title(title_subplots[rule],fontsize=5)
        if x_labels is not None:
            ax[rule].set_xticks(np.arange(len(x_labels)))
            ax[rule].set_xticklabels(x_labels, fontsize=5)
        ax[rule].set_yticks(np.arange(len(y_labels)))
        ax[rule].set_yticklabels(y_labels, fontsize=5)

    camera = Camera(fig)

    # (2) Reuse the figure and after each frame is created, take a snapshot with the camera.
    for t in range(time_steps):
        mark_time = " T=" + str(scale*t)
        if losses is not None and t<len(losses):
            mark_time+=" Loss="+str(round(losses[t],3))#add loss info if given
        ax[0].text(-0.5, -0.1, mark_time, transform=ax[0].transAxes)
        for rule in range(num_rules):
            mask_temp=None
            if mask is not None:
                mask_temp=mask[rule, :,:]
            sns.heatmap(data_np[t, rule, :, :], ax=ax[rule], cmap="YlGnBu", cbar=False,vmin=0, vmax=1, square=True, mask=mask_temp)  # cmap="YlGnBu"
        for rule in range(num_rules):
            if x_labels is not None:
                ax[rule].set_xticks(np.arange(len(x_labels)))
                ax[rule].set_xticklabels(x_labels, fontsize=5)
            if y_labels is not None:
                ax[rule].set_yticks(np.arange(len(y_labels)))
                ax[rule].set_yticklabels(y_labels, fontsize=5, rotation="horizontal")
        camera.snap()

    # (3) After all frames have been captured, create the animation.
    animation = camera.animate()
    path_output=FOLDER_OUTPUT+title_plot+".gif"
    animation.save(path_output)


def param_visu(ax, num_pred, y_labels, title_plot):
    plt.rcParams.update({'font.size': 8})
    plt.ylabel("Predicates", fontsize=6) 
    plt.title(title_plot)
    # X AXIS
    x_labels = ["head", "body1", "body2"]
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=6)
    # YAXIS
    if y_labels == None:
        y_labels=[str(p_i) for i in range(num_pred)]
    y_ticks=[i for i in range(num_pred)]
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=6)


#----------RANDOM TEST VISUALISATION
#data_test=torch.rand((3, 3, 5))
#visualize(data_test, y_labels=["even", "succ", "succ2", "succ3", "succ4"], title_plot="Task Blabla", title_subplots=["rule1", "rule2", "rule3"])
#data_test_temp=torch.rand((5, 4, 3, 7))
#animate(data=data_test_temp, y_labels=["even", "succ", "succ2", "succ3", "succ4", "aux1", "aux2"], title_plot="Task Blabla", title_subplots=["rule0", "rule1", "rule2", "rule3"])
