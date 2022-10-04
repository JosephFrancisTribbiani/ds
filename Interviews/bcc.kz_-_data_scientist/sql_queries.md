Для решения задач использовался синтаксис **PostgreSQL**

### Задача 1

Напишите запрос, который выведет в разрезе стран информацию о кол-ве сотрудников, которые работают на данный момент в компании, и минимальная зарплата должности которых превышает $2000

```SQL
SELECT subquery_3.country_name, count(subquery_3.country_name) AS qty FROM (
	SELECT * FROM (
		SELECT departments.department_id, countries.country_name FROM countries
		JOIN locations ON countries.country_id = locations.country_id
		JOIN departments ON locations.location_id = departments.location_id
		) AS subquery_1
	JOIN (
		SELECT DISTINCT employees.employee_id, employees.department_id FROM employees
		JOIN jobs
		ON employees.job_id = jobs.job_id
		WHERE employees.employee_id NOT IN (SELECT job_history.employee_id FROM job_history)
		AND jobs.min_salary > 2000
		) AS subquery_2
	ON subquery_1.department_id = subquery_2.department_id
	) AS subquery_3
GROUP BY subquery_3.country_name
ORDER BY qty DESC
```


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

Напишите запрос, который выведет информацию в разрезе заказчиков: Имя, Фамилия, **Имя и Фамилия менеджера**, **Должность**, **Департамент**, **Регион**, **Страна**, **Имя и Фамилия менеджера сотрудника**, кол-во совершенных заказов за последние 30 дней, кол-во совершенных заказов за последний 3 месяца, месяц с максимальным кол-вом заказов за последний год, самый популярный заказываемый продукт за последний месяц, кол-во заказов, где стоимость заказа превышает среднюю стоимость всех его заказов за последний год.

```SQL
SELECT DISTINCT customers.cust_first_name, customers.cust_last_name, customers.customer_id, subquery_1.last_30_days, 
subquery_2.last_3_month, subquery_4.best_month, subquery_6.best_product, subquery_8.more_avg_qty FROM customers

LEFT JOIN (
	-- количество заказов за последние 30 дней
	SELECT customer_id, count(customer_id) as last_30_days FROM orders
	WHERE order_date >= date_trunc('day', now()) - interval '29 days'
	GROUP BY customer_id
	) AS subquery_1
ON customers.customer_id = subquery_1.customer_id

LEFT JOIN (
	-- количество заказов за последние 3 месяца
	SELECT customer_id, count(customer_id) as last_3_month FROM orders
	WHERE order_date >= date_trunc('month', now()) - interval '2 month'
	GROUP BY customer_id
	) as subquery_2
ON customers.customer_id = subquery_2.customer_id

LEFT JOIN (
	--месяц с максимальным количеством заказов за последний год
	SELECT DISTINCT ON (subquery_3.customer_id) subquery_3.customer_id, subquery_3.best_month FROM (
		SELECT customer_id, EXTRACT('month' FROM order_date) AS best_month, count(DISTINCT order_id) as qty FROM orders
		WHERE order_date >= date_trunc('year', now())
		GROUP BY customer_id, best_month
		) AS subquery_3
	ORDER BY subquery_3.customer_id ASC, subquery_3.qty DESC
	) AS subquery_4
ON customers.customer_id = subquery_4.customer_id

LEFT JOIN (
	-- самый популярный заказываемый продукт за последний месяц
	SELECT DISTINCT ON (subquery_5.customer_id) subquery_5.customer_id, subquery_5.qty AS best_product FROM (
		SELECT orders.customer_id, order_items.product_id, SUM(order_items.quantity) AS qty FROM orders
		JOIN order_items
		ON orders.order_id = order_items.order_id
		WHERE orders.order_date >= date_trunc('month', now())
		GROUP BY orders.customer_id, order_items.product_id
		) AS subquery_5
	ORDER BY subquery_5.customer_id ASC, subquery_5.qty DESC
	) AS subquery_6
ON customers.customer_id = subquery_6.customer_id

LEFT JOIN (
	-- кол-во заказов, где стоимость заказа превышает среднюю стоимость всех его заказов за последний год
	SELECT subquery_7.customer_id, COUNT(*) AS more_avg_qty FROM (
		SELECT orders.customer_id, orders.order_id, (order_items.unit_price * order_items.quantity) AS order_price, 
		CAST(AVG(order_items.unit_price * order_items.quantity) OVER(PARTITION BY orders.customer_id) AS decimal(6, 2)) AS avg_price FROM orders
		JOIN order_items
		ON orders.order_id = order_items.order_id
		WHERE orders.order_date >= date_trunc('year', now())
		) AS subquery_7
	WHERE subquery_7.order_price > subquery_7.avg_price
	GROUP BY subquery_7.customer_id
	) AS subquery_8
ON customers.customer_id = subquery_8.customer_id
	
ORDER BY customers.customer_id ASC
```
