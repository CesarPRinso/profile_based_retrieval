# InformationRetrieval: Profile-based retrieval

The student is required to create a method based in the space vector model 
to deliver small text snippets to different users depending on their profile. 
For instance, let us suppose that we have 4 different users: the first one being 
interested in politics and soccer, the second in music and films, the third in 
cars and politics and the fourth in soccer alone. An incoming document targeted 
at politics should be delivered to users 1 and 3, while a document on soccer should 
be delivered to users 1 and 4. Students must submit a written report no longer than 
15 pages explaining the method used to encode both the documents and the user 
profiles, together with the algorithm used to process the queries (the more efficient, 
the better). The written report, which is mandatory will provide a grade of 8 
(out of 10 points) maximum. To obtain the maximum grade (10 points out of 10), 
the student must provide a solid implementation of the proposed method in any 
programming language. The instructor recommends students to choose the Python 
programming language or Java since there are plenty of useful code snippets out 
there to help implement the required functionalities. If the student decides to submit 
the optional part, all the required stuff to execute the program must be provided.

## How to run

- go to into the folder of the project
- create and activate a new virtualenv
- then run the following commands

```
$ pip3 install -r requirements.txt
$ python -m textblob.download_corpora
$ python3 main.py
```
