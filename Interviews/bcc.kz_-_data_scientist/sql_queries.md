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

