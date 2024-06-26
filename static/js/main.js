var selectedCategory;
 
function selectCategory() {
    selectedCategory = document.getElementById("category-select").value;
    if (selectedCategory !== "select") {
       //  document.getElementById("user-input-text").style.display = "inline";
       //  document.getElementById("send-button").style.display = "inline";
       appendUserMessage("You have selected the " + selectedCategory + " category.");
    }
}



function sendMessage() {
    var userInput = document.getElementById("user-input-text");
    var message = userInput.value;
    userInput.value = "";
   
    var userMessage = message;
    appendUserMessage(userMessage);
   
    // The rest of your existing fetch logic
    fetch('http://127.0.0.1:5000/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            message: message,
            category: selectedCategory
        })
    })
    .then(response => response.json())
    .then(data => {
        var chatbotMessage = data.message;
        appendBotMessage(chatbotMessage);
    })
    .catch(error => console.error('Error:', error));

}


function appendUserMessage(message) {
    var userDiv = document.createElement('div');
    var userDivSub = document.createElement('div');
    var span = document.createElement('span');
    span.textContent = "You : "
    userDivSub.textContent = message;
    userDiv.classList.add('message-container', 'user-message');
    userDivSub.classList.add('user-message-sub');
    userDiv.appendChild(span);
    userDiv.appendChild(userDivSub);
    document.getElementById('chat-window').appendChild(userDiv);
}

function appendBotMessage(message) {
    var botDiv = document.createElement('div');
    var botDivSub = document.createElement('div');
    var span = document.createElement('span');
    span.textContent = ": Thiksebot"
    botDivSub.textContent = message;
    botDiv.classList.add('message-container', 'bot-message');
    botDivSub.classList.add('bot-message-sub','text-left');
    botDiv.appendChild(span);
    botDiv.appendChild(botDivSub);
    document.getElementById('chat-window').appendChild(botDiv);
}   


        // Chatbot 



  // Get references to the elements
  const chatbotRobort = document.querySelector('.chatbotRobort');
  const chatBlock = document.querySelector('.chatBlock');
  const closebtnX = document.querySelector('.close-x-btn'); 
  const thikseMsg = document.querySelector('.thikse-close-msg')
  const chatBoard = document.querySelector('.company-chatBoard');

  setTimeout(function() {
    chatBoard.classList.add('d-none');
}, 5000); // 500ms matc

  thikseMsg.addEventListener('click', (e)=>  {
    e.stopPropagation()
  
      // Start the transition
      chatBoard.style.opacity = '0';
      chatBoard.style.height = '0';
      chatBoard.style.padding = '0';

      // Wait for the transition to complete before setting display to none
      setTimeout(function() {
          chatBoard.classList.add('d-none');
      }, 500); // 500ms matches the CSS transition duration
  });


  // Add event listener to toggle visibility
  chatbotRobort.addEventListener('click', function() {
    if(chatBlock.classList.contains('d-none')){
      chatBlock.classList.remove('d-none');
      chatBlock.classList.add('d-block');
      
    }
    else{
      chatBlock.classList.add('d-block');
    }
  });
  // Close the chat tab
  closebtnX.addEventListener('click', function(e) {
    e.stopPropagation()
    if(chatBlock.classList.contains('d-block')){
      chatBlock.classList.add('d-none');
      chatBlock.classList.remove('d-block');
    }else{
      chatBlock.classList.add('d-none');
    }

  });