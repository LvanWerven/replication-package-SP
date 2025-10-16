# Research problem
In this document the complete overview of the research problem is stated, including research questions and hypothesis

## Overview research
This study compares the Visual Narrator with the GPT-5 model using a 2 steps 1-shot prompt engineering approach. 

A dataset created and refined in previous study has been used as ground truth. It was further refined such that each class was appointed a category: "must-have" or "should-have". Where a class belongs to the "must-have" category when a instance of the class is manipulated, including create, update, delete. Read as an operation was excluded since reading information does not necessarily mean that the information is dynamic and saved in a database. Created, and synonyms such as submitting, uploadinig etc. are treated with care, since creating a element does also not necessarily mean that it is a class. Only if a property (or attribute) was noted from this element it was labeled as a "must-have" class. 

The task of the visual narrator and GPT-5 is not to identify the classes and their importance (category), there only task is to identify a list of classes. The categories are used to evaluate this identification task.

## Research questions
Main research question: "How well do ChatGPT-5 and Visual Narrator perform in deriving conceptual classes from user stories?"

Sub-research questions: 
* How well do ChatGPT-5 and the Visual Narrator perform in identifying \textit{must-have} classes from user stories?
* How well do ChatGPT-5 and the Visual Narrator perform in identifying \textit{should-have} classes from user stories?

## Hypothesis
Main RQ: Overall identification of conceptual classes
* H<sub>0</sub><sup>0</sup>: There is no significant difference between ChatGPT-5 and the Visual Narrator in their overall performance in identifying conceptual classes.
* H<sub>A</sub><sup>0</sup>: ChatGPT-5 and the Visual Narrator differ significantly in their overall performance in identifying conceptual classes.

RQ1: Identification of must-have classes
* H<sub>0</sub><sup>1</sup>: There is no significant difference between ChatGPT-5 and the Visual Narrator in identifying must-have classes.
* H<sub>A</sub><sup>1</sup>: ChatGPT-5 and the Visual Narrator differ significantly in their ability to identify must-have classes.

RQ2: Identification of should-have classes
* H<sub>0</sub><sup>2</sup>: There is no significant difference between ChatGPT-5 and the Visual Narrator in identifying should-have classes.
* H<sub>A</sub><sup>2</sup>: ChatGPT-5 and the Visual Narrator differ significantly in their ability to identify should-have classes.