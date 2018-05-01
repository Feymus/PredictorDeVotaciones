from sklearn.model_selection import train_test_split
import time
import copy
from Normalizer import Normalizer
from SVMClassifier import SVMClassifier

from tec.ic.ia.pc1.g06 import (
    generar_muestra_pais,
    generar_muestra_provincia
)
from math import log2
import random

class Tree(object):
    "Generic tree node."
    def __init__(self, name='root', children=None):
    
        self.name = name
        self.children = []
        self.votes=[]

        if children is not None:
            for child in children:
                self.add_child(child)
    def __repr__(self):
    	return str(self.name)
    def __str__(self):
    	return str(self.__class__) + ": \n" + str(self.__dict__)
    def add_child(self, node):
    	
    	assert isinstance(node, Tree)
    	self.children.append(node)
    def get_name(self):
    	return self.name
    def get_children(self):
    	return self.children
    def num_children(self):
       	return len(self.children)
    def get_child(self, name):
    	
    	for i in self.children:

    		if(name==i.get_name()):
    			
    			return i
    	
    	
    	return None 
    def delete_child(self, child):
    	self.children.remove(child)
 
    	
    def test(self, test_list, attributes, expected):
    	
    	if(self.name in attributes):
    		
    		position_attr=attributes.index(self.name)
    		attr=test_list[position_attr]
    		child=self.get_child(attr)
    		
    		if(child!=None):
    			children_child=child.children
    			root=children_child[0].name
    			if(isinstance(root, Tree)):
    				return  root.test(test_list, attributes,expected)
    			else:
    				return root
    		else: 
    			
    			return self.children
    		
    		#print(self.children)


    	




  

    
t = Tree('*', [Tree('1'),
               Tree('2'),

               Tree('+', [Tree('3',[Tree('50')]),
                          Tree('4')])])
#print(t.add_child(Tree('35',[])))
#print(t)
#print(t.get_child('3'))
#print(t.test(['+','3'],['*','+']))

#def prunning():


def decision_tree_learning(examples, attributes, parent_examples):

	if not examples:
		
		return plurality_values(parent_examples)
	elif same_classification(examples):
		
		return classifications(examples)
	elif not attributes:
		
		return plurality_values(examples)
	else:

		importance_value=importance(examples)
		exs_attribute=get_column_result(examples, importance_value)
		A= get_attribute(exs_attribute)
		tree=Tree(attributes[importance_value], [])		
		for v in A:
			temp_attr=copy.copy(attributes)
			temp_attr.pop(importance_value)
			exs=get_examples_with_attribute(examples,v,importance_value)
			if(len(temp_attr)+1==len(exs[0])):

				subtree=decision_tree_learning(exs,temp_attr,examples)		
			
				temp_tree=Tree(v,[Tree(subtree,[])])
			
				tree.add_child(temp_tree)
				#tree.add_child(Tree(v,[]))
				#tree.add_child(Tree(subtree,[]))
				


		return tree
		
def same_size (examples):
	size=len(examples[0])
	for i in examples:
		if(len(i)!=size):
			return False
	return True




def get_examples_with_attribute(examples,value,column):
	
	exs=[]
	examples_temp=copy.copy(examples)
	
	for i in examples_temp:


		if (i[column]==value):
			j=copy.copy(i)
			j.pop(column)
			
			exs+=[j]
			
	
	return exs




def get_attribute(results):
	attributes=[]
	for i in results:
		
		attributes.append(i[0])
	return list(set(attributes))


def importance(examples):
	
	if (examples != []):
		max_gain=0
		column=0
		
		for i in range(len(examples[0])-2):
			attribute=get_column_result(examples, i)
			gain_temp=gain(attribute)
			if(max_gain<gain_temp):
				max_gain=gain_temp
				column=i
		return column






def get_column_result(examples, column):
	list_values=[]
	for i in examples:
		value=[i[column]]+[i[len(i)-1]]
		list_values+=[value]
		
	return list_values




def entropy(q):
	sum_q=0
	for i in q:
		sum_q+=i*log2(i)
	return sum_q*-1


def remainder(attribute):
	dic={'total':len(attribute)}
	for i in attribute:
		if (dic.get(i[0])==None):
			
			dic_temp={i[len(i)-1]:1}

			dic.setdefault(i[0],dic_temp)
		else:
			
			temp_dic=dic.get(i[0])
		
			temp_value=temp_dic.get(i[len(i)-1])
			if (temp_value==None):
				dic_temp={i[len(i)-1]:1}

			else:
				dic_temp={i[len(i)-1]:temp_value+1}
			temp_dic.update(dic_temp)

	sum_entropy=0
	
	for key in list(dic.keys())[1:]:
		
		value_temp=sum(list(dic.get(key).values()))/dic.get('total')
	
		entropy_child=value_temp*entropy(list(get_prob(dic.get(key)).values()))
		sum_entropy+=entropy_child

	return sum_entropy
	

def get_count(attribute):
	dic={'total':len(attribute)}
	for i in attribute:
	
		if (dic.get(i[len(i)-1])==None):
			
			dic.setdefault(i[len(i)-1],1)
		else:
	
			temp_value=dic.get(i[len(i)-1]);
			dic_temp={i[len(i)-1]:temp_value+1}
			dic.update(dic_temp)
	return dic
def get_prob(attribute):
	dic_count=get_count(attribute)
	total_values=dic_count.get('total')
	dic_temp={}

	for key in list(dic_count.keys())[1:]:
	
		value=dic_count.get(key)
		prob= value/total_values
		dic_temp.setdefault(key,prob)
	return dic_temp

def gain(attribute):
	
	return entropy(list(get_prob(attribute).values()))-remainder(attribute)






def same_classification(examples):
	if(examples==[]):
		return False
	else:
		votes=classifications(examples)
		if(votes!=[]):
			if(len(votes)==votes.count(votes[0])):
				return True
			else:
				return False


def classifications(examples):
	list_values=[]

	for i in examples:
		size=len(i)
		list_values+=[i[size-1]]
		
	return list_values

def delete_duplicates(values):
	set_values = set(values)

	result = list(set_values)
	return result

def plurality_values(examples):
	votes=classifications(examples)

	classif=delete_duplicates(votes)
	ties=[]
	value_max=0
	classif_max=""
	for i in classif:
		temp=votes.count(i)
		if(temp>value_max):
			value_max=temp
			classif_max=i
			if(ties!=[]):
				ties=[]
		if(temp==value_max):
			ties.append(classif_max)
	if(ties==[]):
		return classif_max
	else:
		return random.choice(ties)
def significance_test(p,n):
	return p/(p+n)

def p_true_irrelevance(p,n,pk,nk):
	return p*((pk+nk)/(p+n))

def n_true_irrelevance(p,n,pk,nk):
	return n*((pk+nk)/(p+n))

'''
def total_deviation(values):
	for i in d:
		chi=
'''

# Our input list.
values = [["Full","no"],["Full","no"],["Some","yes"],["Full","yes"],["Full","no"],["Some","yes"],["None","no"], ["Some","yes"],["Some","yes"],["Full","no"],["None","no"],["Full","yes"]]
values2 = [["French","yes"],["French","no"],["Italian","yes"],["Italian","no"],["Thai","yes"],["Thai","yes"],["Thai","no"],["Thai","no"],["Burger","yes"],["Burger","yes"],["Burger","no"],["Burger","no"]]
values3=[["Full","Thai","no"],["Full","French","no"],["Some","French","yes"],["Full","Thai","yes"],["Full","Italian","no"],["Some","Burger","yes"],["None","Burger","no"], ["Some","Italian","yes"],["Some","Thai","yes"],["Full","Burger","no"],["None","Thai","no"],["Full","Burger","yes"]]
lenData = 6000
samples = generar_muestra_pais(lenData)

lenData1 = 1000
samples_pruning = generar_muestra_pais(lenData1)

attr=[]
for i in range(len(samples[0])-1):
	attr+=["attr"+str(i)]

#print(chi_square(2,5))
tree_test=decision_tree_learning(samples, attr,[])
tree_test.test(samples_pruning[0],attr,samples_pruning[0][len(samples_pruning[0])-1])
#print(tree_test.children)
#print(tree_test.children[0])

fail=0
win=0
for i in samples_pruning:

	result=tree_test.test(i,attr,i[len(i)-1])
	print(result)
	if(i[len(i)-1] in result):

		win+=1
	else:
		fail+=1
print(win)
print(fail)
print((len(samples_pruning)-fail)/len(samples_pruning)*100)
#tree_test.test(samples[0],attr)


