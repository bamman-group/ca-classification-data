# Classification in Cultural Analytics

## data

**Atypical animacy.** Data from Coll Ardanuy et al. 2020 differentiating mentions of machine-related terms (*engine*, *machine*, etc.) as animate vs. non-animate.  Animate machines are those that are depicted as being alive (including people referred to as machines), while inanimate machine are not. We create train/dev/test splits at random.

**Emotion.**  Data from Kim and Klinger 2018, which identifies the emotions that characters are experiencing by tying mentions of characters in a sentence to mentions of a trigger emotion word.  We transform this into a multiclass classification task by extracting sentences in which a character experiences a single emotion, and ask a model to predict the emotion for that character. We create train/dev/test splits so that passages from the same work appear in only one partition.

**Folktales.** Data from Hagedorn et al. 2022, which assigns the ATU type to texts.  We select only ATU types that are attested at least nine times among the text labels, and create train/dev/test splits to maintain the same label distribution across splits (i.e., at least three instances per type across each split).

**Genre.**  We draw inspiration from Sharmaa et al. 2020 in using Library of Congress subject classification as a proxy for genre, and use a subset of 5 genres that work studied (science fiction, detective and mystery stories, adventure stories, love stories, and westerns). We draw texts from Project Gutenberg, sampling 5 passages (each approximately 500 words) from 150 books for each genre.  To enable a multiclass classification problem, we only consider books that are tagged with one subject classification from the set above (so that works that are tagged with both "love stories" and "westerns" are excluded).  We create train/dev/test splits so that texts by the same author appear in only one partition, and we select a maximum of 5 books per author.

**Haiku.** Data from Long and So 2016, which contrasts haiku poems with non-haiku poems. We create train/dev/test splits so that poems by the same author appear in only one partition. 

**Hippocorpus.**  Data from Sap et al. 2020, which solicits first-person stories written by workers on Amazon Mechanical Turk in three categories: *recalled* stories, which narrate real events transpiring within the past six months; *imagined* stories, fictional narratives on the same topic as a randomly selected recalled story; and *retold* stories, recalled stories told again 2-3 months later by the same workers.  This is the only task where a human does not judge a label by inspection of a pre-existing text; accordingly, it is not possible to articulate the textual boundaries between those categories *a priori*. We create train/dev/test splits so that texts by the same author appear in only one partition. 

**Literary time.**  As our sole regression task, we draw data from Underwood 2018, which labels the number of seconds that transpire in a fictional passage of approximately 250 words. We create train/dev/test splits so that texts from the same title appear in only one partition.

**Narrativity.** Data from Piper and Bagga 2022, which contrasts passages from narrative genres (biography, fairy tales, novels) to passages from non-narrative genres (scientific abstracts, book reviews, supreme court proceedings). We create train/dev/test splits so that texts of the same genre appear in only one partition. 

**Strangeness.** Data from Simeone et al. 2017, which contrasts sentences mentioning "descriptions or introductions of technology and novel science" with those that do not, both drawn from Project Gutenberg texts.  We create train/dev/test splits at random.

**Stream-of-consciousness.** Data from Long and So 2016, which contrasts stream-of-consciousness passages with control passages drawn at random from realist novels.  The original work sampled control passages of fixed character lengths, leading to passages that break between words; we re-sample passages from Project Gutenberg texts of the same titles breaking only across sentences, sampling passage lengths to reflect the same empirical distribution of lengths in the SoC texts.  We create train/dev/test splits so that the passages by the same author appear in only one partition. 



## References

This dataset is compiled from sources that have been originally published elsewhere.  Please see the following for the original sources (and please cite when referencing this work).

animacy

```
@inproceedings{coll-ardanuy-etal-2020-living,
    title = "Living Machines: A study of atypical animacy",
	author = "Coll Ardanuy, Mariona and Nanni, Federico and Beelen, Kaspar and Hosseini, Kasra and Ahnert, Ruth and Lawrence, Jon and McDonough, Katherine and Tolfo, Giorgia and Wilson, Daniel CS and McGillivray, Barbara", editor = "Scott, Donia and Bel, Nuria and Zong, Chengqing",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    month = Dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2020.coling-main.400",
    doi = "10.18653/v1/2020.coling-main.400",
    pages = "4534--4545",
}
```

folktales

```
@article{Hagedorn-2022,
 author = {Hagedorn, Joshua and Darányi, Sándor},
 doi = {10.5334/johd.78},
 journal = {Journal of Open Humanities Data},
 keyword = {en_US},
 month = {Jun},
 title = {Bearing a Bag-of-Tales: An Open Corpus of Annotated Folktales for Reproducible Research},
 year = {2022}
}
```

genre

```
@inproceedings{bamman24,
	Author = {David Bamman and Kent K. Chang and Lucy Li and Naitian Zhou},
	Booktitle = {CHR 2024: Computational Humanities Research Conference},
	Title = {On Classification with Large Language Models in Cultural Analytics},
	Year = {2024}
}

```

haiku

```
@article{long2016literary,
  title={Literary pattern recognition: Modernism between close reading and machine learning},
  author={Long, Hoyt and So, Richard Jean},
  journal={Critical inquiry},
  volume={42},
  number={2},
  pages={235--267},
  year={2016},
  publisher={University of Chicago Press Chicago, IL}
}
```

hippocorpus

```
@inproceedings{sap-etal-2020-recollection,
    title = "Recollection versus Imagination: Exploring Human Memory and Cognition via Neural Language Models",
	author = "Sap, Maarten and Horvitz, Eric and Choi, Yejin and Smith, Noah A. and Pennebaker, James", editor = "Jurafsky, Dan and Chai, Joyce and Schluter, Natalie and Tetreault, Joel",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = Jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.acl-main.178",
    doi = "10.18653/v1/2020.acl-main.178",
    pages = "1970--1978",
}
```

literary\_time

```
@article{underwood2018literary,
  title={Why literary time is measured in minutes},
  author={Underwood, Ted},
  journal={ELH},
  volume={85},
  number={2},
  pages={341--365},
  year={2018},
  publisher={Johns Hopkins University Press}
}
```

narrativity

```
@article{piper2022toward,
  title={Toward a data-driven theory of narrativity},
  author={Piper, Andrew and Bagga, Sunyam},
  journal={New Literary History},
  volume={54},
  number={1},
  pages={879--901},
  year={2022},
  publisher={Johns Hopkins University Press}
}
```

reman

```
@inproceedings{kim2018feels,
  title={Who feels what and why? annotation of a literature corpus with semantic roles of emotions},
  author={Kim, Evgeny and Klinger, Roman},
  booktitle={Proceedings of the 27th International Conference on Computational Linguistics},
  pages={1345--1359},
  year={2018}
}
```

soc

```
@article{long2016turbulent,
  title={Turbulent flow: A computational model of world literature},
  author={Long, Hoyt and So, Richard Jean},
  journal={Modern Language Quarterly},
  volume={77},
  number={3},
  pages={345--367},
  year={2016},
  publisher={Duke University Press}
}
```

strangeness

```
@article{simeone2017towards,
  title={Towards a Poetics of Strangeness: Experiments in Classifying Language of Technological Novelty},
  author={Simeone, Michael and Koundinya, Advaith Gundavajhala Venkata and Kumar, Anandh Ravi and Finn, Ed},
  year={2017},
  journal={Journal of Cultural Analytics},
  volume=2,
  issue=1
}
```

