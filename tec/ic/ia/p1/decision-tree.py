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
import numpy as np
import pandas as pd
import scipy.stats as stats

positive=0
negative=0


temp_positive=0

temp_negative=0




class Tree(object):
    "Generic tree node."
    def __init__(self, name='root', children=None):
    
        self.name = name
        self.children = []
        self.gain=0
        self.votes= None
        self.p=0
        self.n=0
        self.set_size=0
        self.set_error=0
        self.pk=0
        self.nk=0

        if children is not None:
            for child in children:
                self.add_child(child)
    def __repr__(self):
    	return str(self.name)
    def __str__(self):

    	return str(self.__class__) + ": \n" + str(self.__dict__)
    def set_data_pruning(self,p,n):
    	#print("POSITIVOS:"+str(p))
    	self.p+=p
    	self.n+=n
    	self.set_size=self.p+self.n
    	self.set_error=(self.set_size-self.n)/self.set_size*100


    def add_child(self, node):
    	
    	assert isinstance(node, Tree)
    	self.children.append(node)
    def set_gain(self,gain):
    	self.gain=gain
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
    def get_leafs(self):
    	
    	votes=[]
    	if(self.votes!=None):
	    	for i in self.votes.keys():
	    		
	    		votes+=[[i]]

    	return Tree(votes,[])


	    	
    def printTree(self):

    	if(self.children!=[]):
    		print("---------------------------------------------------------------------")

    		print(self)
    		for i in self.children:
    			i.printTree()
    	else:
    		print(self)
    		print("---------------------------------------------------------------------")



    def calc_pk_nk(self):
    	#print("WUUU")
    	if(self.children!=[]):
    	
    		
    		for i in self.children:
    			#print("Estoy en el for")
    			#print(self.p + self.n)
    			#print(i.p+i.n)
    			#print(i)
    			if(self.p + self.n>0):
	    			i.pk = self.p*((i.p+i.n)/(self.p+self.n))
	    			i.nk = self.n*((i.p+i.n)/(self.p+self.n))
	    			#print("pk:"+str(i.pk))
	    			#print("nk:"+str(i.nk))
	    			i.calc_pk_nk()




    def test(self, test_list, attributes, expected):
    	global positive
    	global negative
    	global temp_positive
    	global temp_negative
    	if(self.name in attributes):
    		
    		
    		position_attr=attributes.index(self.name)
    		attr=test_list[position_attr]
    		child=self.get_child(attr)
    		
    		if(child!=None):
    			
	    		for i in child.children:
	    			
	    			
	    			if(i.children==[]):
	    				if(expected in i.name):

	    					temp_positive+=1
	    					positive+=1	
	    				else:
	    					temp_negative+=1
	    					negative+=1	
	    				#print("AQUI TOY3")
	    				#print("tpositive"+str(temp_positive))
	    				#print("tnegative"+str(temp_negative))
	    				#i.set_data_pruning(1,0)
	    				#child.set_data_pruning(1,0)
	    				#self.set_data_pruning(1,0)
	    				#print(self)
	    				return (expected in i.name)
	    			else:
	    				
	    				is_correct=i.test(test_list, attributes, expected)
	    				
	    				if(is_correct):
	    					i.set_data_pruning(1,0)
	    					child.set_data_pruning(1,0)
	    					self.set_data_pruning(1,0)
	    					
	    					return is_correct
	    				
	    	
	    		#print("AQUI TOY4")
	    		#print("tpositive"+str(temp_positive))
	    		#print("tnegative"+str(temp_negative))		
	    		child.set_data_pruning(0,1)
	    		self.set_data_pruning(0,1)
	    		#print(child)
	    		return is_correct
	    	else:
	    		#print("AQUI TOY1")
	    		#print("tpositive"+str(temp_positive))
	    		#print("tnegative"+str(temp_negative))
	    		
	    		
	    		temp_negative+=1
	    		negative+=1
	    		self.set_data_pruning(0,1)	
	    		#print(self)		
	    		return False
    	else:
    		#print("AQUI TOY2")
    		#print("tpositive"+str(temp_positive))
	    	#print("tnegative"+str(temp_negative))
    		
    		
    		temp_negative+=1
	    	negative+=1		
	    	self.set_data_pruning(copy.copy(temp_positive),(copy.copy(temp_negative)))
	    	#print(self)
    		return False

    		

 
    	


    '''		
    def pruning(self, threshold, max_gain):
    	#print("INICIO:"+str(self.name))
    	if(len(str(self.name))>4 and str(self.name)[:4]=="attr"):
    		#print("NAME: "+str(self.name))
    		#print("GAIN: "+str(self.gain))
    		for i in self.children:
    			i.pruning(threshold,max_gain)
    			
    	else:
    		
    		for i in self.children:
    		
    			
    			if(isinstance(i.name, str)):
    				#print("---------------------------------------------------------------------------------------")
    			
    				i.pruning(threshold,max_gain)
    				if(i.gain<threshold):
    					

    					
    					if(i.get_leafs!=None):
	    					
	    					self.delete_child(i)
	    					
	    					self.add_child(i.get_leafs())
	    					
    			else:
    				#print("---------------------------------------------------------------------------------------")
    			
    				i.pruning(threshold,max_gain)
    '''
    def chi_square(self):
    	#print("INICIO:"+str(self.name))
    	if(len(str(self.name))>4 and str(self.name)[:4]=="attr"):
    		chi=0
    		for i in self.children:
    			if(i.pk!=0 and i.nk!=0):
    				chi+=(i.p-i.pk)**2/i.pk+(i.n-i.nk)**2/i.nk
    		return chi
    def pruning_chi(self, threshold, attr):


    	if(self.children!=[]):
    		for i in self.children:
    			i.pruning_chi(threshold,attr)
    			chi=self.chi_square()
	    		p=1-threshold
	    		crit = stats.chi2.ppf(p,len(attr)+1) 
	    		
	    		if(chi!=None):
	    			if(chi>crit):
	    				self.delete_child(i)
	    				self.add_child(i.get_leafs())

	    
    	
    	
    



    		










    		
    		


    	




  





def decision_tree_learning(examples, attributes, parent_examples):
	
	if not examples:
		
		return delete_duplicates(plurality_values(parent_examples))
	elif same_classification(examples):
		
		return delete_duplicates(classifications(examples))
	elif not attributes:
		
		return delete_duplicates(plurality_values(examples))
	else:

		importance_value , max_gain=importance(examples)
		exs_attribute=get_column_result(examples, importance_value)
		A= get_attribute(exs_attribute)
		tree=Tree(attributes[importance_value], [])	
		tree.set_gain(max_gain)	
		tree.votes=count_votes(examples)
		for v in A:
			
			temp_attr=copy.copy(attributes)
			temp_attr.pop(importance_value)
			exs=get_examples_with_attribute(examples,v,importance_value)

			if(len(temp_attr)+1==len(exs[0])):

				subtree=decision_tree_learning(exs,temp_attr,examples)
				
				if(isinstance(subtree, Tree)):
					
					temp_tree=Tree(v,[subtree])
				else:
					
				
					temp_tree=Tree(v,[Tree(subtree,[])])
				temp_tree.votes=count_votes(exs)
				tree.add_child(temp_tree)
				
				


				

		
		return tree
def count_votes(examples):
	temp_list=[]
	for i in examples:
		temp_list+=[i[len(i)-1]]
	dic={}
	for i in list(set(temp_list)):
		
		dic.setdefault(i,temp_list.count(i))



	return dic
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
		
		return [column,max_gain]






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



# Our input list.
values = [["Full","no"],["Full","no"],["Some","yes"],["Full","yes"],["Full","no"],["Some","yes"],["None","no"], ["Some","yes"],["Some","yes"],["Full","no"],["None","no"],["Full","yes"]]
values2 = [["French","yes"],["French","no"],["Italian","yes"],["Italian","no"],["Thai","yes"],["Thai","yes"],["Thai","no"],["Thai","no"],["Burger","yes"],["Burger","yes"],["Burger","no"],["Burger","no"]]
values3=[["Full","Thai","no"],["Full","French","no"],["Some","French","yes"],["Full","Thai","yes"],["Full","Italian","no"],["Some","Burger","yes"],["None","Burger","no"], ["Some","Italian","yes"],["Some","Thai","yes"],["Full","Burger","no"],["None","Thai","no"],["Full","Burger","yes"]]
lenData = 5000
samples = generar_muestra_pais(lenData)


lenData1 = 250
samples_pruning = generar_muestra_pais(lenData1)

attr=[]
for i in range(len(samples[0])-1):
	attr+=["attr"+str(i)]


tree_test=decision_tree_learning(samples, attr,[])


#tree_test.test(samples[3],attr,samples[3][len(samples[3])-1])





fail=0
win=0
for i in samples_pruning:

	
	result=tree_test.test(i,attr,i[len(i)-1])

	#print("positive"+str(positive))
	#print("negative"+str(negative))
	#print("tpositive"+str(temp_positive))
	#print("tnegative"+str(temp_negative))
	temp_positive=0
	temp_negative=0
	if(result):
		win+=1
		
	else:
		fail+=1
		
print("---------------------------------------------------------------------")		
print("---------------------------------------------------------------------")
print("SIN PODADO")
print("---------------------------------------------------------------------")
print("ACERTADOS:"+str(win))
print("FALLADOS: "+str(fail))
print("ACCURACY: "+str((len(samples_pruning)-fail)/len(samples_pruning)*100))
print("---------------------------------------------------------------------")
print(tree_test)
tree_test.calc_pk_nk()
tree_test.pruning_chi(0.3,attr)


fail=0
win=0
for i in samples_pruning:

	result=tree_test.test(i,attr,i[len(i)-1])
	if(result):
		win+=1
	else:
		fail+=1
print("---------------------------------------------------------------------")
print("PODADO")
print("---------------------------------------------------------------------")
print("ACERTADOS:"+str(win))
print("FALLADOS: "+str(fail))
print("ACCURACY: "+str((len(samples_pruning)-fail)/len(samples_pruning)*100))
print("---------------------------------------------------------------------")
