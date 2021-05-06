---
title: 'demographicx: A Python package for estimating gender and ethnicity using deep learning transformers'
tags:
  - Python
  - name
  - gender
  - ethnicity
  - race
  - BERT
  - NLP
authors:
  - name: Lizhen Liang
    orcid: 0000-0001-9329-2767
    affiliation: 1 
  - name: Daniel Acuna
    orcid: 0000-0002-7765-1595
    affiliation: 1
affiliations:
 - name: School of Information Studies, Syracuse University
   index: 1

date: 6 May 2021
bibliography: paper.bib

---

# Summary

Plenty of research questions would benefit from understanding whether demographic factors are associated with social phenomena. Accessing this information from individuals is many times infeasible or unethical. While software packages have been developed for inferring this information, they are often untested, outdated, or with licensing restrictions. Here, we present a Python package to infer the gender and ethnicity of individuals using first names or full names. We employ a deep learning transformer of text fragments based on BERT  to fine-tune a network. We train our model on Torkiv [@illinoisdatabankIDB-9087546], and extensively validate our predictions. Our gender prediction achieves an average F1 of 0.942 across female, male, and unknown gender names. Similarly, our ethnicity prediction achieves an average F1 of 0.94 across White, Black, Hispanic, and Asian categories. We hope that by making our package open and tested, we improve demographic estimates for many research fields that are trying to understand these factors.

Demographic information such as gender and ethnicity is a crucial dimension to understand many social phenomena. Gender and ethnicity are of course only a fraction of the critical factors that should be analyzed about individuals (see Acuna [@acuna]), yet they have attracted increased interest from the research community. In social science, for example, it has been shown that gender and race are important for scientific collaboration [@lariviere], mentorship [@schwartz], and funding [@ginther]. Accessing this information is, however, challenging because of legal or ethical reasons. Many studies resort to analyzing names to make these kinds of inferences, but the packages and services they often use are non-reproducible or rely on proprietary information with unknown methods and validations (e.g., genderize.io). Without access to an easy-to-use, public, open, and validated method, we risk making inferences about these kinds of phenomena without good grounding. While inferring demographics from names has potential flaws [@kozlowski], it is sometimes the only input we have; it is desirable to have better algorithms than the ones currently available.



<table>
    <thead>
        <tr>
            <th>       </th>
            <th colspan=2>Male</th>
            <th colspan=2>Female</th>
            <th colspan=2>Unknown</th>
        </tr>
        <tr>
            <th>       </th>
            <th colspan=1>Validation</th>
            <th colspan=1>SSA</th>
            <th colspan=1>Validation</th>
            <th colspan=1>SSA</th>
            <th colspan=1>Validation</th>
            <th colspan=1>SSA</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>F1</td>
            <td>0.961</td>
            <td>0.813</td>
            <td>0.975</td>
            <td>0.915</td>
            <td>0.889</td>
            <td>0.504</td>
        </tr>
        <tr>
            <td>Accuracy</td>
            <td>0.972</td>
            <td>0.711</td>
            <td>0.979</td>
            <td>0.885</td>
            <td>0.862</td>
            <td>0.664</td>
        </tr>
        <tr>
            <td>AUC</td>
            <td>0.993</td>
            <td>0.954</td>
            <td>0.996</td>
            <td>0.965</td>
            <td>0.966</td>
            <td>0.860</td>
        </tr>
    </tbody>
</table> 

  

Here, we describe a Python package called demographicx which infers gender from first name and ethnicity from the full name. It is based on fine-tuning a deep learning BERT embedding model with sub-word tokenization (Devlin et al., 2018). Importantly, our model has the ability to make predictions for names that it has not seen before. We build our package on top of the popular transformers package, which increases the likelihood that users will have parts of our models cached in their computers. The dataset we used to train includes Genni + Ethnea for the Author-ity 2009 dataset by Torvik [@illinoisdatabankIDB-9087546], which has names and predicted results by other previous methods. We mixed the dataset with the Social Security Administration (SSA) popular newborn baby names dataset (Social Security Administration, 2013) and a Wikipedia name ethnicity dataset [@ambekar]. We validate our model with both the aggregated data set and the Wikipedia datasets. Our models achieve excellent performance on both tasks (see tables).

<table>
    <thead>
        <tr>
            <th>       </th>
            <th colspan=2>Black</th>
            <th colspan=2>Hispanic</th>
            <th colspan=2>White</th>
            <th colspan=2>Asian</th>
        </tr>
        <tr>
            <th>       </th>
            <th colspan=1>Validation</th>
            <th colspan=1>Wikipedia</th>
            <th colspan=1>Validation</th>
            <th colspan=1>Wikipedia</th>
            <th colspan=1>Validation</th>
            <th colspan=1>Wikipedia</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>F1</td>
            <td>0.976</td>
            <td>0.987</td>
            <td>0.936</td>
            <td>0.822</td>
            <td>0.907</td>
            <td>0.850</td>
            <td>0.941</td>
            <td>0.859</td>
        </tr>
        <tr>
            <td>Accuracy</td>
            <td>0.999</td>
            <td>0.999</td>
            <td>0.928</td>
            <td>0.788</td>
            <td>0.902</td>
            <td>0.856</td>
            <td>0.931</td>
            <td>0.843</td>
        </tr>
        <tr>
            <td>AUC</td>
            <td>0.999</td>
            <td>0.996</td>
            <td>0.990</td>
            <td>0.964</td>
            <td>0.983</td>
            <td>0.963</td>
            <td>0.989</td>
            <td>0.962</td>
        </tr>
    </tbody>
</table> 

Because our package is built based on the transformers package, it can be easily incorporated into PyTorch and transformers. The API is very simple on purpose. Our package has already been used in Acuna and Liang (2021) and multiple other internal projects.
``` python
In: from demographicx import GenderEstimator
In: gender_estimator = GenderEstimator()
In: gender_estimator.predict(“Daniel”)
Out: {‘female’: 0.001, ‘male’: 0.988, ‘unknown’, 0.011}

In: gender_estimator.predict(“Amy”)
Out: {‘female’: 0.998, ‘male’: 0.001, ‘unknown’, 0.001}

In: from demographicx import EthnicityEstimator
In: ethnicity_estimator = EthnicityEstimator()
In: ethnicity_estimator.predict(“Daniel Acuna”)
Out: {‘white’: 0.002, ‘hispanic’: 0.998, ‘black’, 0.000, ‘asian’: 0.000}

In: ethnicity_estimator.predict(“Lizhen Liang”)
Out: {‘white’: 0.000, ‘hispanic’: 0.000, ‘black’, 0.000, ‘asian’: 0.999}
```

# Acknowledgements

L. Liang and D. Acuna were partially funded by the National Science Foundation grant “Social Dynamics of Knowledge Transfer Through Scientific Mentorship and Publication” #1933803.  

# References