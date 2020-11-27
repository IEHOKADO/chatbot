window.onload = function() {
    run();
};

async function run() {
    var path2model = "https://raw.githubusercontent.com/IEHOKADO/chatbot/master/model/model.json";
    model = await tf.loadLayersModel(path2model);
    var path2idx = ["https://raw.githubusercontent.com/IEHOKADO/chatbot/master/json/char2idx.json",
                    "https://raw.githubusercontent.com/IEHOKADO/chatbot/master/json/idx2char.json"];
                    
    var req = new XMLHttpRequest();
    req.open("GET", path2idx[0], false);
    req.send(null);
    char2idx = JSON.parse(req.responseText);
    req = new XMLHttpRequest();
    req.open("GET", path2idx[1], false);
    req.send(null);
    idx2char = JSON.parse(req.responseText);
}

async function reply() {
    var rawText = $("#textInput").val();
    var userHtml = '<div class="cb userText"><span>' + rawText + '</span></div>';
    $("#textInput").val("");
    $("#chatbox").append(userHtml);
    document.getElementById('point').scrollIntoView({block: 'start', behavior: 'smooth'});
    var botHtml = '<div class="cb botText"><span>' + await predict(rawText) + '</span></div>';
    $("#chatbox").append(botHtml);
    document.getElementById('point').scrollIntoView({block: 'start', behavior: 'smooth'});
}

async function predict(text) {
    var segmenter = new TinySegmenter();
    var segs = segmenter.segment(text);
    var x = [];
    for(var i = 0; i < 15; i++) {
      if(char2idx[segs[i]]) x.push(char2idx[segs[i]]);
      else x.push(0);
    }
    x = tf.tensor2d([x]);
    probas = await model.predict(x);
    var probas_array = await Array.from(await probas.data());
    //console.log(probas_array);
    var max_proba = Math.max.apply(null, probas_array);
    //console.log(max_proba);
    if(max_proba > 0.5) var idx = probas_array.indexOf(max_proba);
    else var idx = 0;
    //console.log(idx);
    var res = idx2char[idx];
    return res;
}

//EnterKeyを押したとき
$("#textInput").keypress(function(e) {
    if ((e.which == 13) && document.getElementById("textInput").value != "" ){
        reply();
    }
})

//SendButtonを押したとき
$("#buttonInput").click(function() {
    if (document.getElementById("textInput").value != "") {
        reply();
    }
})
