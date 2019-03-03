import com.google.gson.Gson;
import java.io.*;
import java.util.HashMap;
import java.util.Hashtable;

class Data {
    private StringBuffer ans= new StringBuffer();
    private HashMap actionMap=new HashMap();

    /**
     * @return  返回解析出来的hashmap
     * @throws IOException
     */
    HashMap input(String readPath) throws IOException {
        ans=new StringBuffer();
        actionMap=new HashMap();
        Gson gson=new Gson();
        File actionFile= new File(readPath);
        InputStreamReader reader=new InputStreamReader(
                new FileInputStream(actionFile));
        BufferedReader bufferedReader=new BufferedReader(reader);
        String tmp = "";
        while(true){
            try {
                if ((tmp =bufferedReader.readLine())==null) {
                    break;
                }
                else{
                    ans.append(tmp);
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        actionMap=gson.fromJson(String.valueOf(ans),actionMap.getClass());
        bufferedReader.close();
        return actionMap;
    }
    void output(Hashtable<String, Long> state, String writePath) throws IOException {
    	Long value = state.get("blood1");
    	if(value < 0) {
    		state.put("blood1", 0L);
    	}
    	value = state.get("blood2");
    	if(value < 0) {
    		state.put("blood2", 0L);
    	}
        state.put("Time",  System.currentTimeMillis());
        Gson gson=new Gson();
        String ret = gson.toJson(state, Hashtable.class);
        /*---------------------*/
        File stateFile=new File(writePath);
        OutputStreamWriter writer=new OutputStreamWriter(
                new FileOutputStream(stateFile));
        BufferedWriter bufferedWriter=new BufferedWriter(writer);
        bufferedWriter.write(ret);
        bufferedWriter.close();
    }
}
