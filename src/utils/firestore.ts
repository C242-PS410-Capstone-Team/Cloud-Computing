import { Firestore } from "@google-cloud/firestore";
import dotenv from "dotenv";

dotenv.config();

// Create a Firestore instance
const firestore = new Firestore({
  projectId: process.env.PROJECT_ID,
  keyFilename: process.env.KEY_FILENAME,
});

const userDbCollection = firestore.collection("users");

export default userDbCollection;
