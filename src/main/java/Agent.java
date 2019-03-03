import java.util.LinkedHashMap;

/**
 * 
 * Copyright (C), 2018-2021, HIT Zkx info. Edu.
 * 
 * FileName: Agent.java
 * 
 * 此类定义了强化深度学习模型中的Agent类
 * 
 * @author 张恺欣
 * @Date 2018/11/26
 * 
 * @version 1.00
 * 
 */
public class Agent {
	long reward = 0;
	/** Agent的id */
	private String id = "";
	/** Agent的状态,按输入顺序排序,非线程安全 */
	private LinkedHashMap<String, Long> state = new LinkedHashMap<>();

	/** Agent的动作列表 */
	/*
	 * final Hashtable action=new Hashtable(){ { //始化action put("NULL",0);
	 * put("goForward",1); put("goBack",2); put("holdHead",3); put("holdBody",4);
	 * put("holdLeg",5); put("attackHead",6); put("attackBody",7);
	 * put("attackLeg",8); } };//无排序，线程安全
	 */

	Agent(String id) {
		/* 初始化id */
		this.id = id;
		/* 初始化state */
		/*
		 * blood Agent的生命值[0,9) sol stand or lie,0-lie,1-stand act Agent的动作,[0,8]，0=NULL
		 * dis Agent在环境中的坐标,[0,8]
		 */
		state.put("blood", 8L);
		// 1为站着
		state.put("sol", 1L);
		state.put("act", 0L);
		state.put("dis", 0L);

	}

	LinkedHashMap getState() {
		return this.state;
	}

	public void setState(String key, long value) {
		this.state.put(key, value);
	}

	public String getId() {
		return this.id;
	}

	long move(Long k) {
		long dis = state.get("dis");
		if (k == 1) {
			if (dis + 1 <= 8) {
				dis += 1;
			}
		} else {
			if (dis - 1 >= 0) {
				dis -= 1;
			}
		}
		state.put("dis", dis);
		return dis;
	}

	/**
	 * @param ag 被攻击的agent 通过攻击获得reward
	 */
	void attack(Agent ag) {
		long act = Long.valueOf(ag.state.get("act"));
		long distance = Math.abs((long) ag.state.get("dis") - (long) this.state.get("dis"));
		reward += this.getHurt(act, ag.state, distance);
		if (ag.judgeDie()) {
			reward += 5;
		}
	}

	/**
	 * 
	 * getHurt详细使用说明
	 * 
	 * 攻击头部成功-2生命值，攻击身体成功-1生命值，攻击腿部成功对面倒地
	 * 
	 * @param act   对手对该Agent施加攻击的类型
	 * 
	 * @param state 施加攻击者的状态，倒下时
	 * 
	 * @return 返回被攻击的Agent被攻击后损失的生命值
	 * 
	 */
	private int getHurt(long act, LinkedHashMap stat, long distance) {
		if(act==6&&distance<=1)
		{
		    //如果没有护头 
			if(this.state.get("act")!=3) {
		    this.state.put("blood", (long) this.state.get("blood") - 2);
			}
		}
		else if(act==7&&distance<=1)
		{
		    if(this.state.get("act")!=4)
		    {
		        this.state.put("blood",(long)this.state.get("blood")-1);
		    }
		}
		else if(act==8&&distance<=1)
		{
		    if(this.state.get("act")!=5)
		    {
		        this.state.put("blood",this.state.get("blood")-1); //倒下
		        this.state.put("sol",0L);
		        //如果此时已倒下 在getHurt函数后会有一个函数通过检测stat_old的值让agent站起来
		    }
		}
		if (distance >= 2) {
			if (this.state.get("act") == 3)
				return -10;
			if (this.state.get("act") == 4)
				return -10;
			if (this.state.get("act") == 5)
				return -10;
			if (this.state.get("act") == 6)
				return -10;
			if (this.state.get("act") == 7)
				return -10;
			if (this.state.get("act") == 8)
				return -10;
		}
		if (this.state.get("blood") <= 3) {
			if (this.state.get("act") == 1 && this.state.get("dis") >= (long) stat.get("dis"))
				return 10;
			if (this.state.get("act") == 2 && this.state.get("dis") <= (long) stat.get("dis"))
				return 10;
			if (this.state.get("act") == 1 && this.state.get("dis") <= (long) stat.get("dis"))
				return -10;
			if (this.state.get("act") == 2 && this.state.get("dis") >= (long) stat.get("dis"))
				return -10;
			if (distance <= 1) {
				if (this.state.get("act") == 0)
					return -10;
				if (this.state.get("act") == 3) {
					if (act == 6)
						return 10;
					else
						return 5;
				}
				if (this.state.get("act") == 4) {
					if (act == 7)
						return 10;
					else
						return 5;
				}
				if (this.state.get("act") == 5) {
					if (act == 8)
						return 10;
					else
						return 5;
				}
				if (this.state.get("act") == 6) {
					if (act == 3)
						return -10;
					else
						return 0;
				}
				if (this.state.get("act") == 7) {
					if (act == 4)
						return -10;
					else
						return 0;
				}
				if (this.state.get("act") == 8) {
					if (act == 5)
						return -10;
					else
						return 0;
				}
			}
		}

		else {
			if (this.state.get("act") == 1 && this.state.get("dis") >= (long) stat.get("dis"))
				return -10;
			if (this.state.get("act") == 2 && this.state.get("dis") <= (long) stat.get("dis"))
				return -10;
			if (this.state.get("act") == 1 && this.state.get("dis") <= (long) stat.get("dis"))
				return 10;
			if (this.state.get("act") == 2 && this.state.get("dis") >= (long) stat.get("dis"))
				return 10;
			if(distance <= 1) {
				if(this.state.get("act") == 6) {
					if(act != 3) return 10;
				}
				if(this.state.get("act") == 7) {
					if(act != 4) return 10;
				}
				if(this.state.get("act") == 8) {
					if(act != 5) return 10;
				}
			}
			if(act==6&&distance<=1)
			{
			    //如果没有护头 
				if(this.state.get("act")==3) {
					return 10;
				}
			}
			else if(act==7&&distance<=1)
			{
			    if(this.state.get("act")==4)
			    {
			        return 10;
			    }
			}
			else if(act==8&&distance<=1)
			{
			    if(this.state.get("act")==5)
			    {
			        return 10;
			    }
			}
		}
		return 0;
	}

	/**
	 * 
	 * holdSelf详细使用说明
	 * 
	 * Agent保护自己的某个位置
	 * 
	 * @param act 对该Agent采取防御的类型
	 * 
	 */
	void holdSelf(Long act) {
		if (act == 3) {
			this.state.put("act", 3L);
			System.out.println(id + ",state:act=" + state.get("act"));
		} else if (act == 4) {
			this.state.put("act", 4L);
			System.out.println(id + ",state:act=" + state.get("act"));
		} else if (act == 5) {
			this.state.put("act", 5L);
			System.out.println(id + ",state:act=" + state.get("act"));
		} else {
			System.out.println("WrongInput(holdSelf)!");
		}
	}

	boolean judgeDie() {
		if (this.state.get("blood") == 0) {
			System.out.println(this.getId() + " blood state: " + this.getState().get("blood"));
			return true;
		} else {
			System.out.println(this.getId() + " blood state: " + this.getState().get("blood"));
			return false;
		}
	}
}
