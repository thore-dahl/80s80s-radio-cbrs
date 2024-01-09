<div align="center">
  <img src="https://github.com/thore-dahl/80s80s-radio-cbrs/assets/130995551/7cb69bd9-bea1-4cd0-8dfc-533db8c63ad0" alt="80s80s-radio" style="width:auto;height:250px;">
  <h1>Content-Based Recommender System (CBRS)</h1>
</div>
<br>
<div align="center">
  <a href="https://flask.palletsprojects.com/en/3.0.x/" target="_blank" rel="noreferrer"> 
    <img src="https://www.vectorlogo.zone/logos/pocoo_flask/pocoo_flask-icon.svg" alt="flask" width="40" height="40"/> 
  </a>
  <a href="https://radimrehurek.com/gensim/" target="_blank" rel="noreferrer"> 
    <img src="https://media.slid.es/uploads/465821/images/5836629/pasted-from-clipboard.png" alt="gensim" width="40" height="40"/> 
  </a>
  <a href="https://matplotlib.org" target="_blank" rel="noreferrer"> 
    <img src="https://raw.githubusercontent.com/valohai/ml-logos/d8dfb916e50a93a41f3b1ed2ca7bd3dbc77030a2/matplotlib.svg" alt="matplotlib" width="40" height="40"/> 
  </a>
  <a href="https://www.nltk.org" target="_blank" rel="noreferrer"> 
    <img src="https://miro.medium.com/v2/resize:fit:592/1*YM2HXc7f4v02pZBEO8h-qw.png" alt="nltk" width="40" height="40"/> 
  </a>
  <a href="https://numpy.org" target="_blank" rel="noreferrer"> 
    <img src="https://www.vectorlogo.zone/logos/numpy/numpy-icon.svg" alt="numpy" width="40" height="40"/> 
  </a>
  <a href="https://pandas.pydata.org/" target="_blank" rel="noreferrer"> 
    <img src="https://raw.githubusercontent.com/devicons/devicon/2ae2a900d2f041da66e950e4d48052658d850630/icons/pandas/pandas-original.svg" alt="pandas" width="40" height="40"/> 
  </a>
  <a href="https://scikit-learn.org/" target="_blank" rel="noreferrer"> 
    <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="scikit_learn" width="40" height="40"/>
  </a>
  <a href="https://spacy.io/" target="_blank" rel="noreferrer"> 
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/SpaCy_logo.svg/1280px-SpaCy_logo.svg.png" alt="spacy" height="20"/> 
  </a>
  <h1></h1>
  A project based on songs' <a href="https://open.spotify.com/intl-de">genres</a>, <a href="https://genius.com">lyrics</a>, and user <a href="https://www.last.fm/de/home">tags</a>. Unstructured data on such songs submitted to NLP methodologies. Preprocessing steps were implemented pursuant to overarching methodologies. Methodologies as manifested in pre-trained fastText for static genre embeddings. As for lyrics, NRCLex for heuristic emotions and sentiments as well as trained Doc2vec for contextualized word embeddings. Contrasting to TF-IDF (count-based) for users' tags. Results were complemented by <a href="https://open.spotify.com/intl-de">sonic</a>. The consequent features were scored by a fusion of cosine similarities and Euclidean distances. That tool was deployed online via Flask, producing Spotify previews on search. Perceptual similarity among those previews improved by 15.56%, relative to the prior tool, as evidenced by surveys. A substitute assistant for <a href="https://www.80s80s.de">80s80s radio</a> music planners and moderators was thus furnished.
</div>
