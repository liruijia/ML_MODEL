-- 河南电力 代扣解约数据
-- 虽然添加了 feedback_msg is not  null 条件，但是 有很多的用户填写的解约原因直接是多个空白符，所以还得进一步的处理
select user_id ,
       cons_no ,
       msg_id ,
       feedback_day,
       feedback_data,
       feedback_code ,
       feedback_msg
from (
select  user_id ,
        cons_no ,
        msg_id ,
        feedback_code ,
        feedback_date ,
        feedback_day,
        feedback_msg,
        ROW_NUMBER() OVER(PARTITION BY msg_id  ORDER BY feedback_date) AS rank1
from  bangdaodata.ods_wh_agreement_feed_back  where ds <= '202100'
and  ds >= '201901'
and pubms_code = 'PWHENAN'
) a
where rank1=1
and   feedback_msg is not null
order by user_id ,cons_no ,msg_id ,feedback_day
limit 10000
;