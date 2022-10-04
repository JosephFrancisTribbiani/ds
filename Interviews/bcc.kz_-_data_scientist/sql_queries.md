### Задача 1


### Задача 2

 Напишите запрос, который выведет информацию о топ 10 заказчиках (Имя, Фамилия, номер телефона), которые сделали совершили больше всего заказов в текущем месяце

```SQL
SELECT customers.cust_first_name, customers.cust_last_name, customers.phone_numbers FROM (
	SELECT customer_id, count(DISTINCT order_id) as norders FROM orders
	WHERE date_trunc('month', order_date) = date_trunc('month', now())
	GROUP BY customer_id
	ORDER BY norders DESC
	) AS subquery
LEFT JOIN customers
ON subquery.customer_id = customers.customer_id
LIMIT 10
```


### Задача 3

