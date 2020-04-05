var el = x => document.getElementById(x);

function showText(){
  text = el('textInput').value;
  el('textOutput').innerHTML = text ;
}

function analyze(){
    text = el('textInput').value;
    if (text.split(" ").length <5) alert('Please select a longer sentence/phrase');

    el('analyze-button').innerHTML = 'Analyzing...';

    var xhr = new XMLHttpRequest();
    var loc = window.location
    xhr.open('POST', `${loc.protocol}//${loc.hostname}:${loc.port}/analyze`,true);
    xhr.onerror = function() {alert (xhr.responseText);}
    xhr.onload = function(e) {
      if (this.readyState === 4) {
        var response = JSON.parse(e.target.responseText);
        el('labelOutput').innerHTML = `Result = ${response['label']}`;
        el('label_prob').innerHTML = `${response['score']}`;
        el('label_confidence').innerHTML = `${response['Confidence']}`;
      }
      el('analyze-button').innerHTML = 'Analyze';
    }

    var formData = new FormData();
    formData.append('unique_name',text)
    xhr.send(formData)
}
