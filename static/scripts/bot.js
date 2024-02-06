fetch('../../data/response.json')
    .then((response) => response.json())
    .then((json) => console.log(json));