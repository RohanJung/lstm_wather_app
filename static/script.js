// const container = document.querySelector('.container');
// const weatherBox = document.querySelector('.weather-box');
// const weatherDetails = document.querySelector('.weather-details');
// const error404 = document.querySelector('.not-found');
//
// // Function to fetch weather data based on latitude and longitude
// const fetchWeatherDataByLocation = (latitude, longitude) => {
//     const APIKey = '2cc781d642e2c9cf47b43675bbc0efb5';
//     fetch(`https://api.openweathermap.org/data/2.5/weather?lat=${latitude}&lon=${longitude}&units=metric&appid=${APIKey}`)
//         .then(response => response.json())
//         .then(json => {
//             handleWeatherData(json);
//         });
// };
//
// // Function to fetch weather data based on provided city
// const fetchWeatherDataByCity = (city) => {
//     const APIKey = '2cc781d642e2c9cf47b43675bbc0efb5';
//     fetch(`https://api.openweathermap.org/data/2.5/weather?q=${city}&units=metric&appid=${APIKey}`)
//         .then(response => response.json())
//         .then(json => {
//             handleWeatherData(json);
//         });
// };
//
// // Function to handle weather data response
// const handleWeatherData = (json) => {
//     if (json.cod == '404') {
//         container.style.height = '450px';
//         weatherBox.classList.remove('active');
//         weatherDetails.classList.remove('active');
//         error404.classList.add('active');
//         return;
//     }
//     container.style.height = '560px';
//     weatherBox.classList.add('active');
//     weatherDetails.classList.add('active');
//     error404.classList.remove('active');
//     const image = document.querySelector('.weather-box img');
//     const temperature = document.querySelector('.weather-box .temperature');
//     const description = document.querySelector('.weather-box .description');
//     const humidity = document.querySelector('.weather-details .humidity span');
//     const wind = document.querySelector('.weather-details .wind span');
//     switch (json.weather[0].main) {
//         case 'Clear':
//             image.src = 'images/clear-new.png';
//             break;
//         case 'Rain':
//             image.src = 'images/rain-new.png';
//             break;
//         case 'Snow':
//             image.src = 'images/snow-new.png';
//             break;
//         case 'Clouds':
//             image.src = 'images/cloud-new.png';
//             break;
//         case 'Mist':
//             image.src = 'images/mist-new.png';
//             break;
//         case 'Haze':
//             image.src = 'images/mist-new.png';
//             break;
//         default:
//             image.src = 'images/clear-new.png';
//     }
//     temperature.innerHTML = `${parseInt(json.main.temp)}°C`;
//     description.innerHTML = `${json.weather[0].description}`;
//     humidity.innerHTML = `${json.main.humidity}%`;
//     wind.innerHTML = `${parseInt(json.wind.speed)}Km/h`;
// };
//
// // Function to handle search button click
// const handleSearch = () => {
//     const city = document.getElementById('search-btn').value;
//     if (city === '') {
//         return;
//     }
//     fetchWeatherDataByCity(city); // Fetch weather data using provided city
// };
//
// // Attach event listener to search button
// document.querySelector('.search-box button').addEventListener('click', handleSearch);
//
// // Check if geolocation is supported
// if (navigator.geolocation) {
//     navigator.geolocation.getCurrentPosition(
//         function(position) {
//             const latitude = position.coords.latitude;
//             const longitude = position.coords.longitude;
//             console.log("Latitude: " + latitude + " Longitude: " + longitude);
//             fetchWeatherDataByLocation(latitude, longitude); // Fetch weather data using obtained coordinates
//         },
//         function(error) {
//             switch(error.code) {
//                 case error.PERMISSION_DENIED:
//                     console.error("User denied the request for Geolocation.");
//                     break;
//                 case error.POSITION_UNAVAILABLE:
//                     console.error("Location information is unavailable.");
//                     break;
//                 case error.TIMEOUT:
//                     console.error("The request to get user location timed out.");
//                     break;
//                 case error.UNKNOWN_ERROR:
//                     console.error("An unknown error occurred.");
//                     break;
//             }
//         }
//     );
// } else {
//     console.error("Geolocation is not supported by this browser.");
// }
