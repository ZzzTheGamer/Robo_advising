window.dccFunctions = window.dccFunctions || {};
window.dccFunctions.sliderDateDisplay = function(oordDate) {
    const displayDate = new Date("0000-12-31");
    displayDate.setDate(displayDate.getDate() + oordDate);
     return displayDate.toDateString();
}
