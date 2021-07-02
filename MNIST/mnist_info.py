def mnist_info():
  sum = 0
  len_y = len(y)
  sum_pct = 0
  nums = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

  print('{:<10} {:<10} {:<10}'.format('digit', 'count', 'percent'))
  print('{:<10} {:<10} {:<10}'.format('------', '-------', '---------'))

  for i in range(10):
    count = len(y[y == str(i)])
    pct = round(count / len_y * 100, 3)
    digits = '{} ({}) :'.format(i, nums[i])
    digits = '{} ({}) :'.format(i, nums[i])
    print('{} ({}) :  {:<5}  {}%'.format(i, nums[i], count, pct))
    sum = sum + count
    sum_pct = sum_pct + pct
  print('total count: ', sum)  
  print('total pct: {} %'.format(round(sum_pct, 2)))

mnist_info()
