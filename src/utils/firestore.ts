import { Firestore } from "@google-cloud/firestore";

const firestoreInstance = new Firestore({
  projectId: "testing-443402",
  keyFilename: "service-key.json",
});
const userDbCollection = firestoreInstance.collection("users");

export default userDbCollection;
