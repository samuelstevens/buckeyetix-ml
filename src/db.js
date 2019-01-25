/*
    This file reads all ticket objects from the database and dumps it as a json
    so that the python file can read it.
*/

// External
const monk = require('monk');
const mongo = require('mongodb');
const fs = require('fs');

// Setup
const db = monk('localhost:27017/BuckeyeTix');

const ticketsCollection = db.get('Tickets');
const usersCollection = db.get('Users');

const getAllTickets = async () => {
  return await ticketsCollection.find({});
};

const saveDataToJson = data => {
  const json = JSON.stringify(data);

  fs.writeFile('/tmp/ml/tickets.json', json, 'utf8', error => {
    if(error) {
      console.log(error);
    }
  });
}

const main = async () => {
  saveDataToJson(await getAllTickets());
  db.close();
};

main();
